# Project 2 Reading Companion — Plain and Simple

*Same format as the Project 1 companion: one paragraph per reading, plain language, no metaphors, no jargon without an immediate definition. Each section ends with its code — here the code already lives in this repo, so sections point at the real files with a short excerpt and a plain comment on what to look for. An equations-and-concepts glossary closes the document. The kernels referenced are working and test-verified on the Spark; the reading explains what they do and why every line is shaped the way it is.*

---

## P L A N — T R I T O N   F A   O N   G B 1 0 ( S I N G L E   G P U )

## C1. The Triton programming model — [CORE — week 1]

**1. Triton tutorial 01–03 (vector add, softmax, matmul).** Triton is a language for writing GPU functions in Python. Its core idea: you write the program for one *block* of the output (say, 128 rows), and the compiler handles the parallel machinery — which threads load which addresses, how loops are unrolled — that CUDA makes you write by hand. The three tutorials teach the whole vocabulary you need: `tl.load`/`tl.store` with masks (reading arrays with edges that don't divide evenly), `tl.arange` (a block of indices), `tl.dot` (a small matrix multiply on a tile), and the grid (how many copies of your program run). Everything in this project is those five things arranged carefully.

**2. Your A1 notes (PMPP ch. 1–5) — re-read the memory chapter only.** The one fact that decides every kernel design: reading from GPU main memory is ~100× slower than doing math on data already in registers. A kernel is fast when it loads each byte once and does a lot of math on it before touching memory again — the ratio of math to bytes is called arithmetic intensity. Attention's problem, and FlashAttention's fix, are both statements about arithmetic intensity.

**3. GB10 specifics (your A1 notes again).** Your GPU is one 48-SM chip sharing 128 GB with the CPU at 273 GB/s. Two consequences for kernels: memory bandwidth is scarcer than on datacenter parts, so avoiding memory traffic matters *more* here, and your absolute TFLOP/s numbers won't transfer — but kernel-vs-kernel ratios will. Same reporting rule as Project 1.

**The code, look for:**
- How the grid in `flash_attention()` maps one program instance to (one block of query rows) × (one batch-head) — `grid = lambda meta: (triton.cdiv(N_Q, meta["BLOCK_M"]), B * H)`.
- How every `tl.load` carries a `mask=` — that is how a kernel survives shapes that don't divide evenly by its block size.

```python
# src/sparklab/p2/kernels/flash_attn.py — the shape of every Triton kernel:
pid_m = tl.program_id(0)        # which slice of the output THIS program owns
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # its row indices
q = tl.load(q_ptrs, mask=offs_m[:, None] < N_Q, other=0.0)  # masked edge-safe load
```

---

## C2. Online softmax — the trick — [CORE — week 1]

**1. Milakov & Gimelshein, "Online normalizer calculation for softmax."** Softmax normally needs three passes over a row: find the max (for numerical safety), sum the exponentials, divide. Three passes means reading the row three times. This paper shows the max and the sum can be maintained *together* in one streaming pass: when a new block raises the running max, multiply the running sum by `exp(old_max - new_max)` to re-express it in the new frame, then add the new block's terms. Same answer, one pass, and — the part that matters for attention — you never need the whole row at once.

**2. The softmax numerics note (any short "why subtract the max" explainer).** `exp(x)` overflows float32 at x ≈ 88. Subtracting the row max before exponentiating keeps every argument ≤ 0, so nothing overflows and the largest term is exactly 1. The online trick is just this safety subtraction maintained incrementally.

**The code, look for:**
- The two-scalar state per row — `m` (running max) and `l` (running sum) — and the one rescale line that is the entire paper.
- Why the kernel makes two passes over the row (compute the pair, then emit) while attention will need only one — attention folds the emit into the value multiply.

```python
# src/sparklab/p2/kernels/online_softmax.py — the entire trick:
m_new = tl.maximum(m, tl.max(x, axis=0))
l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)   # rescale old, add new
m = m_new
```

---

## C3. Naive attention and the wall — [CORE — week 1]

**1. "Attention Is All You Need" — the formula only.** Attention is `softmax(Q Kᵀ / √d) V`: every query row scores itself against every key row (an N×N matrix S), the scores become weights via softmax, and the weights average the value rows. You need nothing else from the paper.

**2. Any FlashAttention explainer's "the problem" section.** The N×N matrix is the wall. At 8k context with 32 heads, S is 8192² × 32 = 2 billion scores *per layer* — materializing it means writing gigabytes to slow memory and reading them straight back for the softmax and the V multiply. The compute is cheap; the memory round-trip is the cost. This is a low-arithmetic-intensity pattern, and the roofline from A1 says such patterns are bandwidth-bound: the GPU's math units idle while bytes crawl.

**The code, look for:**
- `naive_attention()` in `reference.py` materializes S in one line — `s = q @ k.T * scale` — and the comment marks it as "the wall." It is kept forever as the correctness reference *and* the baseline that makes the kernel's point.
- The benchmark caps naive at N ≤ 2048 and reports the cap as a result: "could not run" is the finding.

```python
# src/sparklab/p2/reference.py — the wall, on purpose:
s = (q.float() @ k.float().transpose(-2, -1)) * scale   # (B, H, N, N) — the wall
p = torch.softmax(s, dim=-1)
return (p @ v.float()).to(q.dtype)
```

---

## C4. FlashAttention forward — [CORE — weeks 1–2]

**1. FlashAttention paper (Dao et al.), sections 1–3.** The fix: never build S. Tile it — each program owns a block of query rows, streams the keys/values in column-blocks, computes one small tile of S at a time in fast on-chip memory, folds it into a running output using the online-softmax rescale, and throws the tile away. Memory traffic drops from O(N²) to O(N): each K/V element is read once per query block instead of the whole matrix being written and re-read. The paper's IO analysis is the formal version; the mechanism is exactly C2's trick applied per-row-block with a third running quantity (the output accumulator) rescaled alongside the sum.

**2. FlashAttention-2 paper, section 3 only.** FA2's improvements are scheduling: better work partitioning across thread blocks and fewer non-matmul operations in the inner loop. Read it to know what "the gap" against production kernels is usually made of — not the algorithm, the scheduling. This is close-the-gap ammunition for C6.

**3. Triton "fused attention" tutorial (06).** The same kernel this repo implements, in Triton's own tutorial style. Read it *after* the repo's version: the tutorial adds performance features (persistent programs, TMA on Hopper) that obscure the algorithm on first contact but are exactly the borrowing list for the gap phase.

**The code, look for:**
- The three running quantities per query block — `m_i` (max), `l_i` (sum), `acc` (output) — and the rescale line `acc = acc * alpha[:, None] + tl.dot(p, v)`: C2's trick with the output folded in.
- The causal early-stop: `hi = min((pid_m + 1) * BLOCK_M + (N_K - N_Q), N_K)` — keys after the block's last row can never be visible, so the loop simply ends early. That early end is the entire ~2× causal speedup.
- The GQA line: `h_kv = h // KV_GROUPS` — several query heads read the same K/V head; one integer divide is the whole feature.
- The zero-row guard before the final divide — padding rows that saw no keys must not divide by zero.

```python
# src/sparklab/p2/kernels/flash_attn.py — the inner loop, the whole paper:
qk = tl.dot(q, tl.trans(k)) * sm_scale          # one TILE of S, in registers
m_new = tl.maximum(m_i, tl.max(qk, axis=1))
alpha = tl.exp(m_i - m_new)                     # rescale factor for old state
p = tl.exp(qk - m_new[:, None])
acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)   # fold tile into output
l_i = l_i * alpha + tl.sum(p, axis=1)
m_i = m_new
```

---

## C5. Correctness before speed — [CORE — week 2]

**1. Any "testing numerical code" note + the BF16 format description.** BF16 keeps float32's range but only ~8 bits of precision — values agree to about 2–3 decimal digits. Two consequences: compare kernels against an FP32 reference (never BF16-vs-BF16, which can agree on shared errors), and set tolerances explicitly for BF16 (torch.allclose defaults were tuned for FP32 and will pass real bugs). The repo pins `atol=2e-2, rtol=2e-2` with a comment deriving them.

**2. The masking-bug folklore (any Triton debugging thread).** Kernel bugs cluster at edges: sequence lengths that don't divide by block size, the last block of a causal row, padding rows. Tests must use ragged shapes on purpose — a kernel tested only at powers of two is untested.

**The code, look for:**
- `assert_close()` reports the *worst offender* (value and position) on failure, not just "not close" — the difference between a failing test and a debuggable one.
- The test matrix: seq lens 128/500/1024 (500 is the ragged one), causal × non-causal, GQA at 14:2 and 28:4 — the real Qwen ratios, not round numbers.

```python
# tests/test_p2_kernels.py — ragged on purpose:
@pytest.mark.parametrize("n", [128, 500, 1024])   # masking bugs hide at 500
@pytest.mark.parametrize("causal", [False, True])
```

---

## C6. Benchmarking and the gap — [CORE — weeks 2–3]

**1. `triton.testing.do_bench` source.** Same discipline as Project 1's harness: warmup runs first, many repetitions, GPU-clock timing, a robust central estimate. Never one eyeballed run. It is ~50 lines; read all of them.

**2. SGLang attention-backend docs + FlashInfer repo README.** The production baselines. SGLang dispatches to one of several attention implementations (FlashInfer, its own Triton backend, torch SDPA) based on hardware and flags. On sm_121a (your GB10), FlashInfer's coverage is uneven — which kernel *actually ran* varies. This is why the benchmark's manifest carries a dispatch column: "SDPA beat me by 20%" is meaningless without knowing which SDPA backend executed.

**3. ncu (Nsight Compute) quick-start — three metrics only.** For the close-the-gap phase, three numbers explain most gaps: achieved occupancy (are enough blocks resident to hide memory waits), memory throughput vs peak (are you bandwidth-bound), and the stall breakdown (what warps wait on). Profile your kernel and the winner on the same shape; the largest difference names the technique to borrow.

**4. The panel's close-the-gap rule (chain doc).** A static comparison table is not the artifact. The artifact is: identify why you lose, borrow exactly one technique from the winner (block sizes, pipelining stages, work partitioning), measure the recovered delta. "I found why I lose and recovered 15 of the 40%" is the story.

**The code, look for:**
- Real shapes only in `SHAPES` — Qwen 0.5B is 14 query heads : 2 KV heads at head_dim 64; nothing round.
- The FLOP/s arithmetic: `4 * B*H*N²*D` (two matmuls, two ops per MAC), halved for causal.
- The manifest's `dispatch` block recording torch/triton versions and device — the unciteable-numbers rule from Project 1, applied to kernels.

```python
# src/sparklab/p2/bench_attn.py — shapes from real model configs, never round:
SHAPES = {"qwen2.5-0.5b": (14, 2, 64), "qwen2.5-7b": (28, 4, 128), "llama3-8b": (32, 8, 128)}
```

---

## C7. Portability — [POST-SHIP]

**1. The AMD amendment (chain doc) + Triton-on-ROCm notes.** The kernel uses no NVIDIA-specific operations — `tl.dot`, `tl.load`, `tl.exp` compile for AMD GPUs unchanged. That is stated (one sentence) in the write-up. The stretch: one weekend cloud MI300 run — same kernel, same harness, second vendor — converts the sentence from asserted to demonstrated, and rhymes with the miles-on-AMD story you already know.

---

## Sequencing (critical path)

- **Week 1 [CORE]:** C1–C3 readings against the working code; run the correctness suite yourself inside the pytorch container; then C4 with the kernel open in a second window.
- **Week 2 [CORE]:** C5 (add a shape or tolerance experiment of your own), then C6's first half — run `bench_attn`, get the honest table with the dispatch column.
- **Week 3 [CORE]:** the gap phase: ncu on your kernel and the winner, borrow one technique, measure the delta. That delta paragraph is the write-up's spine. **Ship by Oct 1; fused RMSNorm is the fallback artifact if FA overruns.**
- **[POST-SHIP]:** C7 portability run.

---

## E Q U A T I O N S   &   K E Y   C O N C E P T S

**Attention.** `O = softmax(QKᵀ/√d) V`. Q: (N, d) queries; K, V: (N, d) keys/values per head. S = QKᵀ is (N, N) — the wall.

**Online softmax update.** Streaming blocks with running max m and sum l: `m' = max(m, max(x_block))`, `l' = l·e^(m−m') + Σ e^(x_block−m')`. FA adds the output accumulator: `acc' = acc·e^(m−m') + e^(S_tile−m')·V_tile`.

**Arithmetic intensity.** FLOPs ÷ bytes moved. Naive attention: O(N²d) FLOPs over O(N²) bytes → intensity ~d, bandwidth-bound at practical d. FA: same FLOPs over O(Nd) bytes → intensity ~N, compute-bound. The algorithm's entire value is this ratio change.

**Attention FLOPs.** 4·B·H·N²·d (two matmuls × 2 ops per multiply-add); causal halves it. The benchmark's TFLOP/s divides this by measured time.

**The concepts, one line each.** **Tile** — the small sub-matrix one program instance works on in fast memory. **Program / grid** — one instance of the kernel and the array of all instances launched. **Mask** — per-element load/store validity; how kernels survive ragged edges. **SRAM vs HBM** — on-chip fast memory (tiles live here) vs main GPU memory (the thing FA avoids touching). **Occupancy** — how many blocks are resident per SM to hide memory latency. **Autotune** — trying several (block size, warps, stages) configs and caching the winner per shape. **num_stages** — software pipelining depth: how many tile-loads are in flight ahead of the math. **GQA** — n query heads share one KV head; kernel-side it is one integer divide, memory-side it shrinks KV by the ratio. **Causal early-stop** — ending the key loop at the block's last visible key; the structural half of causal's speedup. **BF16** — float32's range, ~8 bits of precision; compare against FP32, tolerances ~2e-2. **do_bench** — warmup + many reps + robust center; the only accepted timer. **Dispatch column** — recording which backend/kernel variant actually executed; the kernel world's version of "pin the version or the number is unciteable."
