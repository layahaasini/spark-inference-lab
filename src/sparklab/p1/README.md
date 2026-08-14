# P1 — What is prefix caching actually worth for agents?

**Question:** how much does SGLang's RadixAttention save on realistic agentic workloads, in tokens and milliseconds?
**Method:** a parameterized synthetic-trace generator (shared system prefix, append-only per-turn growth, N interleaved sessions, GRPO group mode, Zipf skew) replayed against SGLang on a DGX Spark, cache on vs off, eviction pressure varied via `--mem-fraction-static`.
**Result:** in progress — headline metric is **prefill-tokens-avoided** (the number that translates to dollars), with radix hit rate and p50/p99 TTFT per turn; lands with the September 2026 write-up.

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
