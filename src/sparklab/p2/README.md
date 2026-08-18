# P2 — FlashAttention in Triton vs. the serving stack

**Question:** how close can a hand-written Triton FA kernel get to production attention backends on GB10 — and exactly why does the gap exist?
**Method:** build the attention line (online softmax → naive attention → FA forward: non-causal → causal → GQA), verify against an FP32 reference at explicit BF16 tolerances, benchmark with `triton.testing.do_bench` on real model shapes (Qwen/Llama head counts and GQA ratios), head-to-head vs torch SDPA and SGLang's backends with a dispatched-code-path column, then borrow one technique from the winner and measure the recovered gap.
**Result:** in progress — the Oct 1 artifact. Kernels + correctness suite are in place; gap analysis and close-the-gap phase are the remaining sessions.

**Study guide:** [docs/p2_reading_companion.md](../../../docs/p2_reading_companion.md)

## Pieces

- [kernels/online_softmax.py](kernels/online_softmax.py) — the one trick FA is built on, isolated: stream a row in blocks carrying only (running max, running sum)
- [kernels/flash_attn.py](kernels/flash_attn.py) — FA forward: tiled QK^T, online-softmax rescale, causal early-stop, GQA head mapping; no NVIDIA-specific intrinsics (ROCm-portable by construction)
- [reference.py](reference.py) — naive attention (materializes the N×N wall — keep it, it *is* the baseline), torch SDPA, and explicit BF16 tolerances with worst-offender reporting
- [bench_attn.py](bench_attn.py) — do_bench head-to-head on real shapes, TFLOP/s, manifest with the dispatch column
- [tests/test_p2_kernels.py](../../../tests/test_p2_kernels.py) — every variant vs FP32 reference, ragged seq lens on purpose (masking bugs hide at edges)

## Running (on the Spark, inside the pytorch container)

```bash
docker run --rm --gpus all -v ~/opensource/spark-inference-lab:/ws -w /ws \
    -e PYTHONPATH=/ws/src nvcr.io/nvidia/pytorch:25.12-py3 \
    python -m pytest tests/test_p2_kernels.py -q          # correctness first, always

docker run --rm --gpus all -v ~/opensource/spark-inference-lab:/ws -w /ws \
    -e PYTHONPATH=/ws/src nvcr.io/nvidia/pytorch:25.12-py3 \
    python -m sparklab.p2.bench_attn --causal             # then the numbers
```

## Rigor rules

- No benchmark before its correctness test passes — a fast wrong kernel is a wrong kernel.
- Real shapes only (head_dim 64/128, GQA 14:2 and 28:4) — round numbers flatter kernels and match no deployment.
- Every comparison row names the dispatched code path + version; FlashInfer's sm_121 support is uneven, and "X beat me" is meaningless without which X actually ran.
- The story is the gap analysis, not the table: find why you lose, borrow one technique, measure the delta.
- Fallback (HM amendment): if FA overruns near Oct 1, ship fused RMSNorm as the interim artifact, then finish FA.
