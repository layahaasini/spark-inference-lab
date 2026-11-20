# P1: prefix caching for agent workloads

How much does SGLang's prefix cache (RadixAttention) save on agent-style traffic, and when does it stop working?

**Writeup:** [writeups/p1_prefix_caching.md](../../../writeups/p1_prefix_caching.md)

## Files

- [tracegen.py](tracegen.py): synthetic agent traces (shared system prompt, append-only turns, N interleaved sessions, Zipf scheduling, GRPO group mode)
- [client.py](client.py): one streamed request to SGLang; returns TTFT plus the server's own `prompt_tokens` / `cached_tokens`
- [experiments/prefix_hit.py](experiments/prefix_hit.py): cold vs warm sanity check, run this first
- [experiments/pressure.py](experiments/pressure.py): hit rate and TTFT as the number of sessions grows; writes one JSON per run to `runs/p1/`

## Running (on the Spark)

```bash
bash configs/spark_server.sh                                  # terminal 1: the SGLang server
python -m sparklab.p1.experiments.prefix_hit                  # terminal 2: sanity check
python -m sparklab.p1.experiments.pressure --sessions 2 4 8 16 32
```

To force eviction, shrink the KV pool (`MEM_FRAC=0.05`) and sweep past the predicted knee:

```bash
MEM_FRAC=0.05 bash configs/spark_server.sh
python -m sparklab.p1.experiments.pressure --sessions 16 48 64 72 80 96 128 160
python -m sparklab.p1.experiments.pressure --zipf-alpha 1.1 --max-session-words 5500   # heavy-tail traffic
python -m sparklab.p1.experiments.pressure --group-size 4                              # GRPO rollout shape
```

Cache-off baseline: `bash configs/spark_server.sh --disable-radix-cache`.

## Measurement rules

- Hit rate is summed from the per-request numbers the server reports, not scraped from its counters.
- Every sweep point starts from a flushed cache.
- Warmup goes past `chunked_prefill_size` (8,192 tokens), otherwise the first point's p99 includes one-time startup cost.
- Report p50 and p99, not means.
- Every run saves a manifest (versions, GPU, server args, seeds) with its results.
