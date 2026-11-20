"""Online softmax: the single trick FlashAttention is built on.

Ordinary softmax needs the whole row before it can start (max, then sum,
then divide). The online form streams the row in blocks, carrying only two
scalars per row — the running max m and running sum l — and rescaling the
partial sum whenever a new block raises the max:

    m_new = max(m, max(block))
    l_new = l * exp(m - m_new) + sum(exp(block - m_new))

Same answer, but only one block is held at a time. Inside attention, this is
what lets FlashAttention avoid building the N x N score matrix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(X, Y, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row * stride_row
    y_row = Y + row * stride_row

    m = -float("inf")                       # running max, one scalar per row
    l = 0.0                                 # running sum of exp(x - m)
    # pass 1: stream the row block by block, maintaining (m, l)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float("inf"))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # rescale the old sum into the new max's frame, then add this block
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    # pass 2: emit exp(x - m) / l — a plain elementwise sweep
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float("inf"))
        tl.store(y_row + offs, tl.exp(x - m) / l, mask=offs < n_cols)


def online_softmax(x):
    """Row-wise softmax over the last dim of a 2D tensor, streaming per block."""
    assert x.ndim == 2 and x.is_cuda
    y = torch.empty_like(x)
    _online_softmax_kernel[(x.shape[0],)](x, y, x.shape[1], x.stride(0), BLOCK=1024)
    return y
