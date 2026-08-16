"""Cold-vs-warm sanity check: two requests share a long prefix; the server
itself reports how much of the second one it found cached. Run this first —
if the warm request doesn't show a large cached_tokens count, something in the
setup is wrong and no downstream experiment is worth running."""

import argparse
import random

from ..client import DEFAULT_BASE_URL, flush_cache, generate, warmup
from ..tracegen import TraceConfig, shared_prefix


def main():
    ap = argparse.ArgumentParser(description="Cold vs warm prefix-hit sanity check")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--sys-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    prefix = shared_prefix(TraceConfig(seed=a.seed, sys_tokens=a.sys_tokens))
    salt = random.random()  # unique tag so stale cache entries from old runs can't interfere

    warmup(a.base_url)  # without this, a fresh server's one-time costs inflate the cold number
    flush_cache(a.base_url)
    cold = generate(f"{prefix}\nUSER[{salt}]: turn 1", base_url=a.base_url)
    warm = generate(f"{prefix}\nUSER[{salt}]: turn 2, different suffix", base_url=a.base_url)

    print(f"cold: TTFT {cold.ttft_ms:7.1f} ms | prompt {cold.prompt_tokens} tok | cached {cold.cached_tokens} tok")
    print(f"warm: TTFT {warm.ttft_ms:7.1f} ms | prompt {warm.prompt_tokens} tok | cached {warm.cached_tokens} tok "
          f"({warm.hit_rate:.0%} of prompt matched)")


if __name__ == "__main__":
    main()
