# What is prefix caching actually worth for tool-calling agents?

> **DRAFT — numbers final, prose in progress. Do not publish before the pre-publication checklist at the bottom is cleared.**

**Question:** how much does SGLang's RadixAttention actually save on realistic agentic workloads, in tokens and milliseconds?
**Method:** a parameterized synthetic-trace generator modeling how agent frameworks assemble context (long shared system prefix, append-only per-turn growth, N interleaved sessions, heavy-tail scheduling, GRPO group sampling), replayed against a pinned SGLang server on a DGX Spark — cache on vs off, eviction pressure controlled by shrinking the KV pool.
**Result:** on this workload, prefix caching refunds **~84% of all prefill work** and cuts p50 time-to-first-token ~3.5× (140 ms → 40 ms). The hit rate holds that ceiling until the cache working set crosses pool capacity at a knee **predictable in advance from three measured trace parameters** (predicted 73 sessions, observed 72–80), degrades gradually rather than falling off a cliff, and lands tail-first: p99 TTFT rises 2.5× across the knee while p50 barely moves. Heavy-tail (Zipf) traffic holds an 86.5% hit rate under pressure that drags uniform traffic down to 64.3%, and RL-style group rollouts run at ~96% — the most cache-favorable workload there is.

## Why measure this

Every serving engine now ships prefix caching, and every agent framework produces exactly the workload it loves: a large fixed block of instructions and tool schemas at the top of every request, a conversation that only ever grows at the end, and turn after turn re-sending everything the model has already seen. The qualitative story — "caching helps agents" — is everywhere. What I could not find published: *how much*, measured, on a stated workload, with the failure boundary located. This post is that measurement.

The numbers that translate to money are **prefill tokens avoided** — every cached token is prompt-processing work the GPU never repeats — so that metric leads every table below. Commercial APIs price cached input at roughly 10–25% of the fresh-token rate; an 84% refund at that spread cuts effective prompt-side cost by ~63–76%. *(Re-check and date provider pricing before publication.)*

## Setup

| | |
|---|---|
| Hardware | NVIDIA DGX Spark: GB10 (sm_121a), 128 GB unified LPDDR5X, ~273 GB/s |
| Server | SGLang `0.5.8+7d51a572`, NGC container `nvcr.io/nvidia/sglang:26.02-py3`, driver 580.159.03 |
| Model | Qwen/Qwen2.5-0.5B-Instruct, BF16 (12 KB of KV per token) |
| Launch | `--mem-fraction-static {0.70, 0.05}` `--enable-metrics`, full flags in `configs/spark_server.sh` |
| Instruments | trace generator + measured client + sweep harness in this repo; every run writes a manifest (versions, GPU, clocks, seeds, server args) |

One sentence on scope, once, so the rest of the post doesn't have to hedge: the Spark's memory bandwidth is roughly 30× below datacenter HBM, so **methodology and relative deltas transfer to big hardware; absolute speeds and dollar figures do not.** If anything, expensive prefill relative to memory makes caching economics *more* vivid here, not less.

Ground truth discipline: every hit-rate number below is the sum of per-request `cached_tokens` / `prompt_tokens` reported by the server itself in each response's `meta_info` — never scraped from aggregate counters, never inferred from timing. Timing (TTFT percentiles, measured client-side on a streamed first chunk) is supporting evidence. Every sweep point starts from an explicitly flushed cache. Replay is closed-loop per session by construction — an agent turn cannot exist before the previous response — which is the workload's real shape, not a coordinated-omission artifact; open-loop *session arrival* is future work.

## The workload

The trace generator models what agent frameworks actually do, with each behavior a measured parameter rather than an assumption:

- **Shared head:** one fixed system-prompt + tool-schema block (~3,900 tokens here) starts every session — the same bytes, every request, forever.
- **Append-only growth:** each turn appends a user step and a tool result (~1,040 tokens/turn measured) to the end of the context and edits nothing above it. This invariant is what makes agent traffic cacheable at all, and the generator's tests pin it.
- **Interleaving:** N sessions take turns competing for one cache. Uniform round-robin is the honest hard case; Zipf-weighted scheduling (a few hot sessions dominate) is the realistic case. Both are reported.
- **Group mode:** n samples share one full context and diverge only in a small tail — the GRPO rollout shape used in RL training.
- **Context budget:** sessions stop growing at a configurable limit, the way real frameworks stop at the model's context window. (The first Zipf run taught this the hard way: the hottest session outgrew `context_len` and the server rightly refused it.)

Tokens are approximated as words at generation time; every published count is the server's own tokenizer-true report at replay time.

## Results

### 1. Cache off vs cache on: the refund

Same trace (2–32 sessions × 5 turns), same server config, one flag flipped:

| sessions | cache-off p50 / p99 TTFT | cache-on p50 / p99 TTFT | hit rate | tokens avoided |
|---|---|---|---|---|
| 2 | 148.8 / 227† ms | 41.9 / 112.7 ms | 79.5% | 55,826 |
| 8 | 136.2 / 217.4 ms | 38.2 / 106.3 ms | 83.8% | 237,856 |
| 32 | 139.0 / 250.6 ms | 41.0 / 68.1 ms | 83.9% | 993,825 |

**~3.5× on p50 TTFT, ~84% of prefill work refunded.** Both lines are flat in session count — correctly so, since below cache capacity neither depends on concurrency. The ~84% is this trace shape's *attainable maximum*, not a deficiency: each turn's freshly appended tool result is new to the world and must be prefilled once; only the shared head and prior history can ever hit.

† First-point p99 excluded as a known warmup artifact; see Honesty section.

### 2. The knee: predicting where caching stops working

Shrink the pool to force the interesting regime: at `--mem-fraction-static 0.05` the server reports a **383,544-token** KV pool at startup (the server's printed capacity, not my arithmetic). The working-set model says the cache is full when

```
shared head + sessions × per-session history  =  pool
3,900      + n × 5,200                        =  383,544   →   n ≈ 73
```

— both trace parameters measured from the baseline run, prediction written down before the sweep ran. Measured, uniform scheduling:

| sessions | p50 TTFT | p99 TTFT | hit rate | tokens avoided |
|---|---|---|---|---|
| 48 | 39.6 ms | 68.4 ms | 84.2% | 1,491,719 |
| 64 | 40.7 ms | 73.2 ms | 84.6% | 1,932,930 |
| 72 | 39.9 ms | 107.3 ms | 84.0% | 2,189,287 |
| 80 | 42.8 ms | 159.6 ms | 81.6% | 2,321,106 |
| 96 | 46.8 ms | 172.9 ms | 75.6% | 2,644,441 |
| 128 | 51.4 ms | 188.6 ms | 67.8% | 3,182,906 |
| 160 | 55.8 ms | 187.4 ms | 64.3% | 3,709,701 |

Ceiling through 64, first crack at 72–80, prediction validated within ~10%. Two properties of the collapse matter more than its location:

- **It is a slope, not a cliff.** Even at 160 sessions — working set 2.2× the pool — the hit rate is 64%, not 6%. Under LRU with append-only growth, every miss immediately re-primes the evicted branch, which then survives long enough to serve the session's next turns. Eviction and reuse reach a moving equilibrium instead of thrashing to zero.
- **It lands tail-first.** Across the knee, p99 TTFT rises 2.5× (68 → 173 ms) while p50 moves ~15%. Most requests still hit; the unlucky ones pay full re-prefill. An agent that runs ten sequential turns samples the tail with probability ~1 − 0.99¹⁰ ≈ 10% — for agents, **the p99 is the contract**, and the knee shows up there first.

### 3. Traffic shape: the honest case vs the realistic case

Same pool, same session counts, Zipf-weighted scheduling (α = 1.1, context-capped):

| sessions | uniform hit | Zipf hit | Zipf tokens avoided |
|---|---|---|---|
| 64 | 84.6% | 89.6% | 3,062,019 |
| 96 | 75.6% | 89.1% | 4,651,994 |
| 160 | **64.3%** | **86.5%** | 7,641,090 |

At maximum pressure, heavy-tail traffic holds a **22-point hit-rate advantage**. The mechanism is classical cache theory landing exactly as predicted: LRU keeps hot sessions resident, hot sessions carry most of the traffic, and evictions concentrate on cold sessions nobody revisits. Real agent fleets are heavy-tailed, so uniform-traffic benchmarks — including most published ones — *understate* what prefix caching is worth in production. (Comparison caveat: capped Zipf sessions grow longer than uniform ones, so the two curves have different attainable maxima; normalize each to its own ceiling before plotting them together.)

### 4. Group rollouts: the RL preview

GRPO-style group sampling — 4 samples sharing each context, diverging only at the tail:

| sessions | p50 TTFT | hit rate | tokens avoided |
|---|---|---|---|
| 16 | 22.1 ms | 95.9% | 2,191,284 |
| 64 | 25.5 ms | 96.1% | 8,953,955 |
| 96 | 21.8 ms | 93.9% | **12,976,262** |

**~96% hit rate at the lowest latency of any workload measured** — three of every four requests replay a context the first sample just primed. This is the shape RL training produces during rollouts, and it means the rollout phase of multi-turn RL is nearly free on the prompt side when the cache is sized right. Measuring this *inside* an actual training run is the bridge experiment of Project 3.

## What this means for capacity

The standard capacity identity: if prefill is a fraction *f* of serving work and *h* of it is refunded, sustainable traffic multiplies by 1 / (1 − f·h). At the measured h ≈ 0.84 and a Mooncake-typical f ≈ 0.55, that projects **~1.86× more traffic on the same hardware** — before counting the queueing bonus (Little's law: shorter requests → fewer in flight → less waiting near saturation, so the delivered improvement exceeds the time saving alone). *f* was not measured in this campaign; validating the projection against goodput-under-SLO curves is listed as future work, and the formula's breakdown condition (f·h → 1) is nowhere near this regime.

## Honesty section: what these numbers are not

- **The first number I got was wrong, and the cache-off baseline caught it.** My cold/warm sanity check initially showed a 123× TTFT ratio. The cache-off sweep then showed full re-prefill of mid-size contexts costs ~140 ms — so a 19.6k-token prefill cannot legitimately cost 3.8 s. That first-ever request had absorbed one-time server warmup; the honest ratio for that prompt is roughly an order of magnitude. Running the weakened-baseline check from Heiser's measurement-crimes list is the only reason this didn't ship.
- **A residual warmup artifact survives:** the first sweep point after any server restart carries one ~3 s outlier in its p99 even after a representative-size warmup request; the suspect is the chunked-prefill code path (engaged only above 8,192 prompt tokens), which the warmup didn't reach. Affected cells are marked; fix (a warmup request above `chunked_prefill_size`) is queued.
- **Absolute numbers stay home.** Ratios, curve shapes, the knee-prediction method, and the traffic-shape gap transfer; milliseconds and dollars from a 273 GB/s single node do not.
- **One model, one engine, serial replay.** 0.5B-class model, SGLang only, one request in flight (the per-session closed loop is real; fleet-level concurrent arrivals are not modeled). vLLM's block-hash design would show the same shape with coarser matching granularity — measuring that is the obvious cross-engine follow-up.

## Reproducing

Every table row traces to a JSON in `runs/` carrying its full manifest (software versions, GPU, clocks, seeds, server args, environment hash). The repo's tests pin the generator's invariants — determinism, append-only growth, shared head, group sharing, context caps — plus two server-gated end-to-end checks. One server flag, one command per experiment:

```bash
bash configs/spark_server.sh                    # add --disable-radix-cache for the off baseline
python -m sparklab.p1.experiments.pressure --sessions 16 48 64 72 80 96 128 160
```

## Pre-publication checklist

- [ ] Fix warmup to exceed `chunked_prefill_size`; re-run first sweep points; remove † markers
- [ ] Re-run cold/warm sanity check post-warmup-fix; quote the honest ratio
- [ ] Charts: hit-rate-vs-sessions (uniform + Zipf + predicted-knee line), TTFT p50/p99 pair, tokens-avoided bars
- [ ] Normalize uniform/Zipf curves to per-shape attainable maxima
- [ ] Re-check cached-token pricing pages; date the citation
- [ ] Every chart re-checked against the measurement-crimes list
- [ ] Final pass: does every starred claim appear as a plain sentence a screener can quote?
