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


def _sample(name, labels):
    from prometheus_client import REGISTRY
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _stub_ok(text, base_url=None, max_new_tokens=8, timeout=None):
    # mirrors the real generate() signature so a changed call site in app.py
    # fails here instead of only in production
    return GenResult(ttft_ms=10.0, latency_ms=50.0, prompt_tokens=100, cached_tokens=80, text="hi")


def test_generate_ok_counts(monkeypatch):
    monkeypatch.setattr(ui, "generate", _stub_ok)
    before = _sample("sparklab_ui_requests_total", {"status": "ok"})
    cached_before = _sample("sparklab_ui_tokens_total", {"kind": "cached"})
    r = client.post("/api/generate", json={"prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hi" and body["cached_tokens"] == 80
    # exact deltas, not substring presence — catches double-increments and
    # wrong-label bugs that "label exists somewhere" checks cannot
    assert _sample("sparklab_ui_requests_total", {"status": "ok"}) == before + 1
    assert _sample("sparklab_ui_tokens_total", {"kind": "cached"}) == cached_before + 80


def test_generate_error_counts(monkeypatch):
    def boom(text, base_url=None, max_new_tokens=8, timeout=None):
        raise ConnectionError("server down")
    monkeypatch.setattr(ui, "generate", boom)
    before_err = _sample("sparklab_ui_requests_total", {"status": "error"})
    before_ok = _sample("sparklab_ui_requests_total", {"status": "ok"})
    r = client.post("/api/generate", json={"prompt": "hello"})
    assert r.status_code == 502
    assert "server down" in r.json()["error"]
    assert _sample("sparklab_ui_requests_total", {"status": "error"}) == before_err + 1
    assert _sample("sparklab_ui_requests_total", {"status": "ok"}) == before_ok


def test_rejects_empty_prompt():
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
