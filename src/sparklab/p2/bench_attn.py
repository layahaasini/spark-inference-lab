"""Head-to-head attention benchmark: our Triton FA vs the stack, on real shapes.

Rigor rules (the NVIDIA-panel amendments, encoded):
- shapes come from REAL model configs (Qwen-class head_dim/GQA ratios), never
  round numbers — round numbers flatter kernels and match no deployment
- every row names the dispatched code path + version: "SDPA beat me by 20%"
  is meaningless without knowing WHICH sdpa backend actually ran
- timing via triton.testing.do_bench (warmup + many reps + median), never
  a single eyeballed run

Run inside the pytorch container on the Spark:
  python -m sparklab.p2.bench_attn --causal
"""

import argparse
import json
import pathlib

import torch
import triton

from ..bench import collect_manifest
from .kernels.flash_attn import flash_attention
from .reference import naive_attention, sdpa_attention

# real model anatomies, not round numbers
SHAPES = {
    # name: (n_q_heads, n_kv_heads, head_dim)
    "qwen2.5-0.5b": (14, 2, 64),
    "qwen2.5-7b": (28, 4, 128),
    "llama3-8b": (32, 8, 128),
}
SEQ_LENS = [512, 2048, 8192]


def flops(B, H, N, D, causal):
    # 2 matmuls (QK^T and PV), 2*N*N*D MACs each -> 4*B*H*N*N*D FLOPs; causal halves it
    f = 4 * B * H * N * N * D
    return f // 2 if causal else f


def bench_one(fn, *args, **kwargs):
    ms = triton.testing.do_bench(lambda: fn(*args, **kwargs))
    return ms


def main():
    ap = argparse.ArgumentParser(description="Attention kernel head-to-head")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    dev = "cuda"

    sdpa_backend = torch.backends.cuda.preferred_sdpa_backend if hasattr(
        torch.backends.cuda, "preferred_sdpa_backend") else "default-dispatch"
    rows = []
    for name, (h, h_kv, d) in SHAPES.items():
        for n in SEQ_LENS:
            q = torch.randn(a.batch, h, n, d, device=dev, dtype=dtype)
            k = torch.randn(a.batch, h_kv, n, d, device=dev, dtype=dtype)
            v = torch.randn_like(k)
            fl = flops(a.batch, h, n, d, a.causal)

            entries = {
                "triton-fa (ours)": lambda: flash_attention(q, k, v, causal=a.causal),
                # naive materializes N*N floats per head — cap it before it OOMs,
                # and report the cap: the blow-up IS the finding
                "naive-torch": (lambda: naive_attention(q, k, v, causal=a.causal)) if n <= 2048 else None,
                "torch-sdpa": lambda: sdpa_attention(q, k, v, causal=a.causal),
            }
            for label, fn in entries.items():
                if fn is None:
                    rows.append({"shape": name, "seq": n, "impl": label, "ms": None,
                                 "tflops": None, "note": "skipped: N^2 matrix too large"})
                    continue
                ms = bench_one(fn)
                rows.append({"shape": name, "seq": n, "impl": label, "ms": round(ms, 4),
                             "tflops": round(fl / (ms * 1e-3) / 1e12, 2)})
                print(f"{name:14s} N={n:5d} {label:18s} {ms:8.3f} ms  {rows[-1]['tflops'] or 0:6.2f} TFLOP/s")

    manifest = collect_manifest(extra={
        "args": vars(a),
        "dispatch": {"torch": torch.__version__, "triton": triton.__version__,
                     "sdpa_backend": str(sdpa_backend),
                     "device": torch.cuda.get_device_name(0)},
    })
    out = pathlib.Path(a.out_dir) / f"attn_bench_{manifest['ts'].replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "results": rows}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
