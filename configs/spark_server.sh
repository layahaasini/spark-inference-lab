#!/usr/bin/env bash
# Pinned SGLang server launch for all Project 1 experiments.
# Any change to these settings must be recorded alongside the numbers it produced;
# the run manifest captures the sglang version and this file captures the flags.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
# sizes the weights+KV pool out of the Spark's one shared memory;
# the sweep variable for eviction-pressure experiments
MEM_FRAC="${MEM_FRAC:-0.70}"

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRAC" \
    --enable-metrics \
    "$@"  # e.g. --disable-radix-cache for the cache-off baseline
