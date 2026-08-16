"""Minimal inference web UI with its own Prometheus metrics.

A thin FastAPI proxy in front of the SGLang server: serves one static page,
forwards prompts through the measured client, and exposes /metrics so UI
traffic shows up in Prometheus/Grafana alongside the server's own numbers.
Binds to localhost only — access is by ssh tunnel, like Grafana.

Run:  uvicorn sparklab.ui.app:app --host 127.0.0.1 --port 8080
"""

import pathlib

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..p1.client import generate

# the custom metrics: request count by outcome, end-to-end latency, and the
# same prompt/cached token accounting the experiments use — so the UI's cache
# behavior is comparable with every other number in the project
REQUESTS = Counter("sparklab_ui_requests_total", "Inference requests from the web UI", ["status"])
LATENCY = Histogram("sparklab_ui_request_seconds", "End-to-end UI request latency (seconds)")
TOKENS = Counter("sparklab_ui_tokens_total", "Prompt-side tokens for UI requests", ["kind"])

app = FastAPI(title="sparklab inference UI")


@app.get("/metrics")
def metrics():
    # plain route, not a mounted sub-app: bare /metrics must answer 200 directly
    # (a mount 307-redirects to /metrics/, which not every scraper follows)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class GenRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)


@app.get("/")
def index():
    return FileResponse(pathlib.Path(__file__).parent / "static" / "index.html")


@app.post("/api/generate")
def api_generate(req: GenRequest):
    try:
        with LATENCY.time():
            r = generate(req.prompt, max_new_tokens=req.max_new_tokens)
    except Exception as e:  # server down, timeout, refused — count it, say why
        REQUESTS.labels(status="error").inc()
        return JSONResponse({"error": str(e)}, status_code=502)
    REQUESTS.labels(status="ok").inc()
    TOKENS.labels(kind="prompt").inc(r.prompt_tokens)
    TOKENS.labels(kind="cached").inc(r.cached_tokens)
    return {
        "text": r.text,
        "ttft_ms": round(r.ttft_ms, 1),
        "latency_ms": round(r.latency_ms, 1),
        "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens,
        "hit_rate": round(r.hit_rate, 4),
    }
