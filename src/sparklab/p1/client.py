"""Measured SGLang client. One request, three numbers.

Ground truth for cache behavior is the server's own report — prompt_tokens and
cached_tokens in the final stream chunk's meta_info. Timing (TTFT) is
supporting evidence only: it moves with batching, scheduling, and thermals,
while the token counts do not.
"""

import json
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")


@dataclass
class GenResult:
    ttft_ms: float          # time to first stream chunk
    latency_ms: float       # full request wall time
    prompt_tokens: int      # server-reported prompt size
    cached_tokens: int      # server-reported tokens found in the radix cache

    @property
    def hit_rate(self):
        return self.cached_tokens / max(self.prompt_tokens, 1)


def generate(text, base_url=DEFAULT_BASE_URL, max_new_tokens=8, timeout=300):
    """Send one streaming request; return timing plus the server's cache report.
    max_new_tokens defaults small because Project 1 measures the prompt side."""
    t0 = time.perf_counter()
    ttft = None
    last = b""
    with requests.post(f"{base_url}/generate", stream=True, timeout=timeout, json={
        "text": text,
        "stream": True,
        "sampling_params": {"max_new_tokens": max_new_tokens},
    }) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1e3
            last = line  # the final chunk carries meta_info with the token counts
    latency = (time.perf_counter() - t0) * 1e3
    meta = json.loads(last.removeprefix(b"data: "))["meta_info"]
    return GenResult(
        ttft_ms=ttft,
        latency_ms=latency,
        prompt_tokens=meta.get("prompt_tokens", 0),
        cached_tokens=meta.get("cached_tokens", 0),
    )


def flush_cache(base_url=DEFAULT_BASE_URL):
    """Empty the radix cache so the next request is truly cold."""
    requests.post(f"{base_url}/flush_cache", timeout=30).raise_for_status()


def server_info(base_url=DEFAULT_BASE_URL):
    """The server's own description of its settings — goes into every run manifest."""
    r = requests.get(f"{base_url}/get_server_info", timeout=30)
    r.raise_for_status()
    return r.json()
