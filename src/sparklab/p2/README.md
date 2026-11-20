# P2: FlashAttention in Triton

How close can a hand-written Triton FlashAttention kernel get to FlashInfer and torch SDPA on GB10, and why is there a gap?

**Writeup:** [writeups/p2_flash_attention.md](../../../writeups/p2_flash_attention.md)

## Files

- [kernels/online_softmax.py](kernels/online_softmax.py): softmax computed block by block with a running max and sum (the trick FlashAttention is built on)
- [kernels/flash_attn.py](kernels/flash_attn.py): FlashAttention forward with causal masking and GQA, plus the split loop and base-2 softmax from the profiling round
- [reference.py](reference.py): naive attention (builds the full N×N matrix), torch SDPA wrapper, and BF16 tolerances
- [bench_attn.py](bench_attn.py): benchmark against naive, SDPA, FlashInfer and SGLang's Triton backend; writes `runs/p2/attn_bench_<timestamp>.json`
- [profile_attn.py](profile_attn.py): runs one kernel a few times so Nsight Compute can profile it
- [tests/test_p2_kernels.py](../../../tests/test_p2_kernels.py): every variant vs an FP32 reference, including lengths that don't divide into blocks

## Running (on the Spark, inside a container with Triton)

```bash
docker run --rm --gpus all -v ~/opensource/spark-inference-lab:/ws -w /ws \
    -e PYTHONPATH=/ws/src nvcr.io/nvidia/pytorch:25.12-py3 \
    python -m pytest tests/test_p2_kernels.py -q          # correctness first

docker run --rm --gpus all -v ~/opensource/spark-inference-lab:/ws -w /ws \
    -e PYTHONPATH=/ws/src nvcr.io/nvidia/sglang:26.02-py3 \
    python -m sparklab.p2.bench_attn --causal             # SGLang image includes FlashInfer
```

Profiling (needs `--cap-add SYS_ADMIN` on the container):

```bash
ncu --set detailed --launch-count 3 --kernel-name-base demangled \
    -k regex:fa_fwd -o runs/p2/ncu_ours \
    python -m sparklab.p2.profile_attn --impl ours --shape qwen2.5-7b --seq 8192
```
