"""Correctness for the P2 kernels: every variant vs an FP32 reference at
explicit BF16 tolerances. Skips cleanly without a CUDA GPU + Triton (runs on
the Spark, inside the pytorch container)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

pytestmark = pytest.mark.needs_gpu

from sparklab.p2.kernels.flash_attn import flash_attention          # noqa: E402
from sparklab.p2.kernels.online_softmax import online_softmax       # noqa: E402
from sparklab.p2.reference import assert_close, naive_attention     # noqa: E402


def test_online_softmax_matches_torch():
    x = torch.randn(64, 1000, device="cuda", dtype=torch.float32) * 8
    assert_close(online_softmax(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-4,
                 label="online softmax")


# non-multiples of block sizes on purpose: masking bugs hide at ragged edges
@pytest.mark.parametrize("n", [128, 500, 1024])
@pytest.mark.parametrize("causal", [False, True])
def test_fa_matches_reference(n, causal):
    torch.manual_seed(0)
    q = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ref = naive_attention(q, k, v, causal=causal)
    assert_close(flash_attention(q, k, v, causal=causal), ref,
                 label=f"fa n={n} causal={causal}")


@pytest.mark.parametrize("h,h_kv,d", [(14, 2, 64), (28, 4, 128)])  # real GQA ratios
def test_fa_gqa(h, h_kv, d):
    torch.manual_seed(1)
    q = torch.randn(1, h, 512, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, h_kv, 512, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ref = naive_attention(q, k, v, causal=True)
    assert_close(flash_attention(q, k, v, causal=True), ref, label=f"gqa {h}/{h_kv}")
