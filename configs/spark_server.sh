#!/usr/bin/env bash
# Pinned SGLang server launch for all Project 1 experiments.
# Runs inside NVIDIA's NGC container so the entire software environment is the
# pin: cite the image tag (plus the server's own /get_server_info report, which
# the run manifest captures) with every published number.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/sglang:26.02-py3}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
# sizes the weights+KV pool out of the Spark's one shared memory;
# the sweep variable for eviction-pressure experiments
MEM_FRAC="${MEM_FRAC:-0.70}"

# host networking so localhost:PORT works; HF cache mounted so models persist
exec docker run --rm --gpus all --network host \
    --name sparklab-server \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRAC" \
    --enable-metrics \
    "$@"  # e.g. --disable-radix-cache for the cache-off baseline
