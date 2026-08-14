"""End-to-end checks that need a live SGLang server. They skip automatically
when no server answers at SGLANG_BASE_URL (default http://localhost:30000), so
the suite is runnable anywhere and strict only where it can be."""

import random

import pytest
import requests

from sparklab.p1.client import DEFAULT_BASE_URL, flush_cache, generate
from sparklab.p1.tracegen import TraceConfig, shared_prefix

pytestmark = pytest.mark.needs_server


def _server_up():
    try:
        return requests.get(f"{DEFAULT_BASE_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.fixture(autouse=True)
def require_server():
    if not _server_up():
        pytest.skip(f"no SGLang server at {DEFAULT_BASE_URL}")


def test_warm_faster_than_cold():
    prefix = shared_prefix(TraceConfig(seed=0, sys_tokens=4000))
    salt = random.random()
    flush_cache()
    cold = generate(f"{prefix}\nUSER[{salt}]: A")
    warm = generate(f"{prefix}\nUSER[{salt}]: B")
    # a RATIO, not a fixed ms number — absolute speeds don't transfer across machines
    assert warm.ttft_ms < 0.7 * cold.ttft_ms
    # and the server itself must report a real match, the actual ground truth
    assert warm.cached_tokens > 0.5 * warm.prompt_tokens


def test_flush_makes_cold():
    prefix = shared_prefix(TraceConfig(seed=0, sys_tokens=2000))
    salt = random.random()
    flush_cache()
    generate(f"{prefix}\nUSER[{salt}]: A")
    flush_cache()
    again = generate(f"{prefix}\nUSER[{salt}]: A")
    # after a flush the same prompt must be a miss — guards silent flush failures
    assert again.cached_tokens < 0.1 * again.prompt_tokens
