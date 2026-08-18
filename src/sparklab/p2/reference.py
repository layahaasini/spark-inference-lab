"""Reference attention implementations and correctness tolerances.

The naive version materializes the full N x N score matrix — that is the
memory wall the whole project exists to feel: O(N^2) bytes written and read
back, versus FlashAttention's O(N) working set. Keep it: it is the baseline
that makes the kernel's point.

Tolerance discipline: the kernel computes in BF16 inputs with FP32
accumulation, so it is compared against an FP32 reference, with tolerances
set for BF16's ~3 decimal digits — explicit numbers, not torch.allclose
defaults (which were tuned for FP32-vs-FP32 and hide real bugs at BF16).
"""

import torch

# BF16 has ~8 bits of mantissa: values agree to ~2-3 decimal digits.
# atol covers the near-zero regime, rtol the bulk of the distribution.
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2


def naive_attention(q, k, v, causal=False):
    """Materialize S = QK^T in full, softmax it, multiply by V. Shapes
    (B, H, N, D). GQA handled by repeating KV heads to match Q heads."""
    if k.shape[1] != q.shape[1]:  # GQA: expand kv heads for the reference
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = q.shape[-1] ** -0.5
    s = (q.float() @ k.float().transpose(-2, -1)) * scale   # (B, H, N, N) — the wall
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril_()
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return (p @ v.float()).to(q.dtype)


def sdpa_attention(q, k, v, causal=False):
    """PyTorch's fused scaled_dot_product_attention — the production torch path."""
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def assert_close(got, ref, atol=BF16_ATOL, rtol=BF16_RTOL, label=""):
    """Explicit-tolerance comparison that reports the worst offender on failure."""
    diff = (got.float() - ref.float()).abs()
    bound = atol + rtol * ref.float().abs()
    bad = diff > bound
    if bad.any():
        i = diff.argmax()
        raise AssertionError(
            f"{label}: {int(bad.sum())}/{bad.numel()} elements out of tolerance; "
            f"worst |diff|={diff.flatten()[i]:.4e} at ref={ref.float().flatten()[i]:.4e}")
