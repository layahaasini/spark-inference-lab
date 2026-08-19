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

try:  # the production backend under test; lives in the SGLang container
    import flashinfer
except ImportError:
    flashinfer = None

try:  # SGLang's own Triton prefill backend (varlen layout, GQA-aware)
    from sglang.srt.layers.attention.triton_ops.prefill_attention import context_attention_fwd
except ImportError:
    context_attention_fwd = None

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
            if flashinfer is not None and a.batch == 1:
                # flashinfer's single-request prefill wants NHD layout (seq, heads, dim);
                # convert once outside the timed region — layout is a setup cost here
                q_fi = q[0].permute(1, 0, 2).contiguous()
                k_fi = k[0].permute(1, 0, 2).contiguous()
                v_fi = v[0].permute(1, 0, 2).contiguous()
                entries["flashinfer"] = lambda: flashinfer.single_prefill_with_kv_cache(
                    q_fi, k_fi, v_fi, causal=a.causal)
            if context_attention_fwd is not None and a.batch == 1:
                # sglang's triton backend wants varlen (total_tokens, H, D) with
                # per-sequence boundaries; single sequence -> start=[0], len=[n]
                q_sg = q[0].permute(1, 0, 2).contiguous()
                k_sg = k[0].permute(1, 0, 2).contiguous()
                v_sg = v[0].permute(1, 0, 2).contiguous()
                o_sg = torch.empty_like(q_sg)
                b_start = torch.zeros(1, dtype=torch.int32, device=dev)
                b_len = torch.full((1,), n, dtype=torch.int32, device=dev)

                def _sglang_triton():
                    context_attention_fwd(q_sg, k_sg, v_sg, o_sg, b_start, b_len, n,
                                          is_causal=a.causal)
                    return o_sg
                entries["sglang-triton"] = _sglang_triton
            for label, fn in entries.items():
                if fn is None:
                    rows.append({"shape": name, "seq": n, "impl": label, "ms": None,
                                 "tflops": None, "note": "skipped: N^2 matrix too large"})
                    continue
                try:
                    ms = bench_one(fn)
                except Exception as e:
                    # on sm_121a a backend refusing a shape IS a result — record it
                    rows.append({"shape": name, "seq": n, "impl": label, "ms": None,
                                 "tflops": None, "note": f"FAILED: {type(e).__name__}: {str(e)[:200]}"})
                    print(f"{name:14s} N={n:5d} {label:18s}   FAILED ({type(e).__name__})")
                    continue
                rows.append({"shape": name, "seq": n, "impl": label, "ms": round(ms, 4),
                             "tflops": round(fl / (ms * 1e-3) / 1e12, 2)})
                print(f"{name:14s} N={n:5d} {label:18s} {ms:8.3f} ms  {rows[-1]['tflops'] or 0:6.2f} TFLOP/s")

    manifest = collect_manifest(extra={
        "args": vars(a),
        "dispatch": {"torch": torch.__version__, "triton": triton.__version__,
                     "sdpa_backend": str(sdpa_backend),
                     "flashinfer": getattr(flashinfer, "__version__", None) if flashinfer else "not installed",
                     "device": torch.cuda.get_device_name(0)},
    })
    out = pathlib.Path(a.out_dir) / f"attn_bench_{manifest['ts'].replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "results": rows}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
