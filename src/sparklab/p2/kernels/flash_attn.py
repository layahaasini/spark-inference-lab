"""FlashAttention forward in Triton: non-causal, causal, and GQA.

The idea in one paragraph: attention wants softmax(QK^T)V, but the score
matrix S is N x N — at 8k context that is 64M floats per head, far too big to
keep anywhere fast. So tile it: each program owns one BLOCK_M-row slice of Q,
streams K/V in BLOCK_N-column tiles, and folds each tile into a running
output using the online-softmax rescale (see online_softmax.py). S never
exists; the working set is a few tiles in SRAM.

Two changes made after profiling with ncu (see writeups/p2_flash_attention.md):

- SPLIT LOOP: the causal mask only bites on the diagonal band. The key loop
  runs in two phases — an unmasked body covering tiles that are fully visible
  (no per-element compares, no bounds masks, cheaper loads), then a short
  masked phase for the diagonal/ragged tail. At 8k, ~95% of tiles take the
  cheap path.
- BASE-2 SOFTMAX: exp2 instead of exp, with the softmax scale pre-multiplied
  by log2(e). Mathematically identical (exp(x) == exp2(x*log2e)); hardware-
  cheaper because the SFU computes exp2 natively and the scale multiply is
  folded into the QK product.

Portability note: nothing here is NVIDIA-specific — no inline PTX, no
warp-level intrinsics — so the same kernel compiles for ROCm.

Layout: (B, H, N, D) row-major. GQA: Q has H heads, K/V have H_KV <= H heads;
query head h reads kv head h // (H // H_KV). D (head_dim) must be a power of
two <= 256 (BLOCK_D == D, one tile covers the full head dim).
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # log2(e)


@triton.jit
def _inner(acc, l_i, m_i, q, K, V, kv_base_k, kv_base_v,
           stride_kn, stride_kd, stride_vn, stride_vd,
           offs_m, offs_d, qk_scale, causal_limit, N_K,
           lo, hi,
           MASKED: tl.constexpr, IS_CAUSAL: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Stream key tiles in [lo, hi). MASKED=False is the hot path: tiles known
    fully visible and fully in-bounds, so no compares and no load masks."""
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + kv_base_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + kv_base_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_K, other=0.0)
        else:
            k = tl.load(k_ptrs)
        # qk arrives already in log2 space: qk_scale folds sm_scale * log2(e)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if MASKED:
            qk = tl.where(offs_n[None, :] < N_K, qk, -float("inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_n[None, :] <= causal_limit[:, None], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        if MASKED:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_K, other=0.0)
        else:
            v = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["N_K", "BLOCK_D", "IS_CAUSAL"],
)
@triton.jit
def _fa_fwd(Q, K, V, O, qk_scale,
            stride_qb, stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_vd,
            stride_ob, stride_oh, stride_om, stride_od,
            H, KV_GROUPS, N_Q, N_K,
            IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)                  # which BLOCK_M slice of queries
    pid_bh = tl.program_id(1)                 # which (batch, q-head)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // KV_GROUPS                     # GQA: several q heads share one kv head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_Q, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kv_base_k = b * stride_kb + h_kv * stride_kh
    kv_base_v = b * stride_vb + h_kv * stride_vh
    # query at row m (aligned to the END of the kv sequence) sees keys <= m + (N_K - N_Q)
    causal_limit = offs_m + (N_K - N_Q)

    if IS_CAUSAL:
        # visibility of the block's FIRST row bounds the fully-visible region;
        # its LAST row bounds where any work remains at all
        hi_total = tl.minimum(pid_m * BLOCK_M + BLOCK_M + (N_K - N_Q), N_K)
        hi_unmasked = tl.maximum((pid_m * BLOCK_M + (N_K - N_Q) + 1) // BLOCK_N * BLOCK_N, 0)
        hi_unmasked = tl.minimum(hi_unmasked, hi_total)
    else:
        hi_total = N_K
        hi_unmasked = (N_K // BLOCK_N) * BLOCK_N   # complete tiles; ragged tail is masked

    acc, l_i, m_i = _inner(acc, l_i, m_i, q, K, V, kv_base_k, kv_base_v,
                           stride_kn, stride_kd, stride_vn, stride_vd,
                           offs_m, offs_d, qk_scale, causal_limit, N_K,
                           0, hi_unmasked,
                           MASKED=False, IS_CAUSAL=IS_CAUSAL,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    acc, l_i, m_i = _inner(acc, l_i, m_i, q, K, V, kv_base_k, kv_base_v,
                           stride_kn, stride_kd, stride_vn, stride_vd,
                           offs_m, offs_d, qk_scale, causal_limit, N_K,
                           hi_unmasked, hi_total,
                           MASKED=True, IS_CAUSAL=IS_CAUSAL,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)

    # rows that saw no keys (padding rows past N_Q) would divide by zero
    acc = acc / tl.where(l_i[:, None] == 0.0, 1.0, l_i[:, None])

    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_Q)


def flash_attention(q, k, v, causal=False):
    """FlashAttention forward. q: (B, H, N_q, D); k, v: (B, H_kv, N_k, D) with
    H a multiple of H_kv (GQA). Returns (B, H, N_q, D) in q's dtype."""
    B, H, N_Q, D = q.shape
    _, H_KV, N_K, _ = k.shape
    assert H % H_KV == 0, "Q heads must be a multiple of KV heads (GQA)"
    assert D in (16, 32, 64, 128, 256), "head_dim must be a power of two <= 256"
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N_Q, meta["BLOCK_M"]), B * H)
    _fa_fwd[grid](
        q, k, v, o, D ** -0.5 * LOG2E,   # softmax scale folded with log2(e)
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, H // H_KV, N_Q, N_K,
        IS_CAUSAL=causal, BLOCK_D=D,
    )
    return o
