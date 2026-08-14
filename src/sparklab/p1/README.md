# P1 — What is prefix caching actually worth for agents?

**Question:** how much does SGLang's RadixAttention save on realistic agentic workloads, in tokens and milliseconds?
**Method:** a parameterized synthetic-trace generator (shared system prefix, append-only per-turn growth, N interleaved sessions, GRPO group mode, Zipf skew) replayed against SGLang on a DGX Spark, cache on vs off, eviction pressure varied via `--mem-fraction-static`.
**Result (measured, write-up pending):** caching refunds **~84% of prefill work** on this trace shape and buys ~3.5× on p50 TTFT vs `--disable-radix-cache`. On a deliberately shrunk 383k-token pool, hit rate holds its ceiling until the working-set knee — predicted at 73 sessions from measured trace parameters, observed at 72–80 — then degrades gradually (64.3% at 2.2× pool, even scheduling). Heavy-tail (Zipf) traffic holds 86.5% under the same pressure; GRPO group-shape rollouts reach ~96%. Full write-up September 2026.

## Pieces

- [tracegen.py](tracegen.py) — the trace generator (shared module; Project 3 reuses it for rollout traces)
- [client.py](client.py) — measured request: TTFT plus the server's own `prompt_tokens` / `cached_tokens` report, which is the ground truth for every hit-rate number
- [experiments/prefix_hit.py](experiments/prefix_hit.py) — cold/warm sanity check; run before anything else
- [experiments/pressure.py](experiments/pressure.py) — hit rate and TTFT vs session-count pressure, one JSON per run with its manifest

## Running (on the Spark)

```bash
bash configs/spark_server.sh                      # terminal 1: pinned server settings
python -m sparklab.p1.experiments.prefix_hit      # terminal 2: sanity check
python -m sparklab.p1.experiments.pressure --sessions 2 4 8 16 32
```

## Rigor rules

- SGLang version pinned; every run writes a manifest (versions, GPU, clocks, settings, seeds). Unpinned numbers are unciteable.
- Hit rate is summed from per-request server reports, never scraped from aggregate counters.
- Distributions, not means: p50/p99 everywhere; a 10-turn agent hits the p99 with ~10% probability.
- Report both even (honest) and Zipf (realistic) session scheduling.
- Reporting rule: the Spark's 273 GB/s memory is ~30× below datacenter HBM — methodology and relative deltas transfer; absolute numbers do not.
