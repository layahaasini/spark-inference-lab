"""Memory-pressure sweep: replay traces with growing session counts and watch
the hit rate fall as the working set stops fitting the cache pool.

Method notes:
- every sweep point starts from a flushed cache, so points are comparable
- hit rate is the sum of per-request server-reported numbers, never a scrape
  of the server's aggregate counters
- replay is closed-loop by construction: an agent turn cannot exist before the
  previous turn's response, so waiting between a session's turns is the
  workload, not a coordinated-omission artifact. (Open-loop arrival of *new*
  sessions is future work and will be stated explicitly when added.)
- vary eviction pressure from the server side via --mem-fraction-static in
  configs/spark_server.sh; record it in the manifest for every run
"""

import argparse
import json
import pathlib

import requests

from ...bench import collect_manifest, percentiles
from ..client import DEFAULT_BASE_URL, flush_cache, generate, server_info
from ..tracegen import TraceConfig, gen_trace, shared_prefix


def run_point(cfg, base_url):
    """One sweep point: fresh cache, full trace replay, distribution summary."""
    try:
        flush_cache(base_url)
    except requests.HTTPError:
        pass  # a --disable-radix-cache server has nothing to flush
    rows = [generate(ev["prompt"], base_url=base_url) for ev in gen_trace(cfg)]
    prompt = sum(r.prompt_tokens for r in rows)
    cached = sum(r.cached_tokens for r in rows)
    return {
        "n_sessions": cfg.n_sessions,
        "requests": len(rows),
        "ttft_ms": percentiles([r.ttft_ms for r in rows]),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_rate": cached / max(prompt, 1),
        "tokens_avoided": cached,  # the headline: prefill work the cache refunded
    }


def main():
    ap = argparse.ArgumentParser(description="Cache hit rate vs session-count pressure sweep")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--sessions", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--sys-tokens", type=int, default=800)
    ap.add_argument("--group-size", type=int, default=1)
    ap.add_argument("--zipf-alpha", type=float, default=0.0)
    ap.add_argument("--max-session-words", type=int, default=None,
                    help="per-session word budget; required in zipf mode so hot sessions stay under context_len")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args()

    manifest = collect_manifest(extra={"server_info": server_info(a.base_url), "args": vars(a)})
    # untimed throwaway at representative size: the first LARGE prefill after a
    # restart pays one-time costs that a tiny warmup prompt does not trigger,
    # and they would otherwise land in the first point's p99 (seed chosen so
    # this prompt shares nothing with any trace)
    generate(shared_prefix(TraceConfig(seed=10**6, sys_tokens=a.sys_tokens)), base_url=a.base_url)
    results = []
    for n in a.sessions:
        cfg = TraceConfig(seed=a.seed, n_sessions=n, turns=a.turns, sys_tokens=a.sys_tokens,
                          group_size=a.group_size, zipf_alpha=a.zipf_alpha,
                          max_session_words=a.max_session_words)
        res = run_point(cfg, a.base_url)
        results.append(res)
        t = res["ttft_ms"]
        print(f"sessions={n:3d}  p50={t['p50']:7.1f}ms  p99={t['p99']:7.1f}ms  "
              f"hit={res['hit_rate']:5.1%}  tokens-avoided={res['tokens_avoided']:,}")

    out = pathlib.Path(a.out_dir) / f"pressure_{manifest['ts'].replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "results": results}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
