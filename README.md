# spark-inference-lab

Systems experiments on a DGX Spark (GB10, 128 GB unified memory): inference measurement, kernels, and agentic RL. One repo, one shared benchmarking module — every experiment imports from `sparklab.bench`.

| # | Project | Question | Status |
|---|---------|----------|--------|
| 1 | [Prefix-caching economics](src/sparklab/p1/README.md) | What is prefix caching actually worth for tool-calling agents? | in progress |
| 2 | FlashAttention in Triton vs. the serving stack | How close can a hand-written Triton FA kernel get to production attention backends on GB10, and why does the gap exist? | planned |
| 3 | Tool-Loop Gym: agentic RL on one box | Can the full agentic-RL loop run on a single Spark, and what does prefix caching do to its economics? | planned |

**Reporting rule (applies to every write-up):** the Spark's 273 GB/s LPDDR5X is roughly 30× below datacenter HBM. Methodology and relative deltas transfer to big hardware; absolute numbers do not.

## Layout

```
src/sparklab/bench/   shared benchmarking module: percentile timing, run manifests
src/sparklab/p1/      Project 1: trace generator, measured client, experiments
configs/              pinned server launch scripts
tests/                pytest suite (server-dependent tests skip when no server is up)
runs/                 experiment outputs, one JSON per run, manifest included
```

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest              # server-dependent tests skip automatically
```

Experiments run against an SGLang server started with `configs/spark_server.sh` (pinned settings; record the exact SGLang version with every result — unpinned numbers are unciteable).
