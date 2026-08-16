"""UI proxy tests: page serves, metrics expose, requests count. No SGLang
server involved — generate() is stubbed, so these run anywhere the ui extras
are installed (and skip cleanly where they aren't)."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import sparklab.ui.app as ui
from sparklab.p1.client import GenResult

client = TestClient(ui.app, raise_server_exceptions=False)


def test_index_serves():
    r = client.get("/")
    assert r.status_code == 200
    assert "sparklab inference" in r.text


def test_generate_ok_counts(monkeypatch):
    monkeypatch.setattr(ui, "generate", lambda prompt, max_new_tokens: GenResult(
        ttft_ms=10.0, latency_ms=50.0, prompt_tokens=100, cached_tokens=80, text="hi"))
    r = client.post("/api/generate", json={"prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hi" and body["cached_tokens"] == 80
    metrics = client.get("/metrics").text
    assert 'sparklab_ui_requests_total{status="ok"}' in metrics
    assert 'sparklab_ui_tokens_total{kind="cached"}' in metrics


def test_generate_error_counts(monkeypatch):
    def boom(prompt, max_new_tokens):
        raise ConnectionError("server down")
    monkeypatch.setattr(ui, "generate", boom)
    r = client.post("/api/generate", json={"prompt": "hello"})
    assert r.status_code == 502
    assert "server down" in r.json()["error"]
    assert 'sparklab_ui_requests_total{status="error"}' in client.get("/metrics").text


def test_rejects_empty_prompt():
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
