"""FlashAttention forward in Triton: non-causal, causal, and GQA.

The idea in one paragraph: attention wants softmax(QK^T)V, but the score
matrix S is N x N — at 8k context that is 64M floats per head, far too big to
keep anywhere fast. So tile it: each program owns one BLOCK_M-row slice of Q,
streams K/V in BLOCK_N-column tiles, and folds each tile into a running
output using the online-softmax rescale (see online_softmax.py). S never
exists; the working set is a few tiles in SRAM. That is the entire paper.

Portability note: nothing here is NVIDIA-specific — no inline PTX, no
warp-level intrinsics, just tl.dot/tl.load/tl.exp — so the same kernel
compiles for ROCm. Stated in the write-up; nearly free in Triton.

Layout: (B, H, N, D) row-major. GQA: Q has H heads, K/V have H_KV <= H heads;
query head h reads kv head h // (H // H_KV). D (head_dim) must be a power of
two <= 256 (BLOCK_D == D, one tile covers the full head dim).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    ],
    key=["N_K", "BLOCK_D", "IS_CAUSAL"],
)
@triton.jit
def _fa_fwd(Q, K, V, O, sm_scale,
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

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running row sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)        # running output

    # causal: rows in this block attend to at most key index (row position);
    # keys strictly beyond the block's last row can never be visible, so the
    # stream stops early — that early stop is where causal's ~2x savings lives
    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M + (N_K - N_Q), N_K)
    else:
        hi = N_K

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_K, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale                  # (BLOCK_M, BLOCK_N) tile of S
        qk = tl.where(offs_n[None, :] < N_K, qk, -float("inf"))
        if IS_CAUSAL:
            # query at position m (aligned to the END of the kv sequence)
            # sees keys up to m + (N_K - N_Q)
            causal_ok = offs_n[None, :] <= (offs_m[:, None] + (N_K - N_Q))
            qk = tl.where(causal_ok, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)                             # rescale factor for old state
        p = tl.exp(qk - m_new[:, None])

        v_ptrs = V + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_K, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

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
        q, k, v, o, D ** -0.5,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, H // H_KV, N_Q, N_K,
        IS_CAUSAL=causal, BLOCK_D=D,
    )
    return o
