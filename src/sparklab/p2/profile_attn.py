"""Minimal repro for ncu: run one attention implementation a few times at one
shape, so the profiler has clean launches to attach to. Usage (in-container):

  ncu --set detailed --launch-count 3 --kernel-name-base demangled \
      -k regex:fa_fwd -o /ws/runs/ncu_ours \
      python -m sparklab.p2.profile_attn --impl ours --shape qwen2.5-7b --seq 8192
"""

import argparse

import torch

from .bench_attn import SHAPES
from .kernels.flash_attn import flash_attention
from .reference import sdpa_attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ours", "sdpa"], default="ours")
    ap.add_argument("--shape", default="qwen2.5-7b", choices=sorted(SHAPES))
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--causal", action="store_true", default=True)
    a = ap.parse_args()

    h, h_kv, d = SHAPES[a.shape]
    q = torch.randn(1, h, a.seq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, h_kv, a.seq, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fn = flash_attention if a.impl == "ours" else sdpa_attention

    fn(q, k, v, causal=a.causal)          # warmup + autotune outside profiling
    torch.cuda.synchronize()
    for _ in range(a.iters):              # the launches ncu captures
        fn(q, k, v, causal=a.causal)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
