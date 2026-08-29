# Project 2 Reading Companion — Plain and Simple

*One short paragraph per reading, an analogy where it helps, then a pointer into this repo's working code. The kernels are already built and test-verified on the Spark — you're reading to understand them, not to build them. Glossary at the end.*

---

## C1. How Triton works — [CORE — week 1]

**The idea in one line:** a GPU is thousands of workers; Triton lets you write instructions for *one small team* of them in Python, and it clones your instructions across all the teams.

**1. Triton tutorials 01–03.** You write a program that handles one *block* of the output — say, 64 rows. Triton runs many copies of it side by side, each copy handling its own block. You only ever need five words of vocabulary: `tl.load` (fetch numbers), `tl.store` (write numbers back), `tl.arange` (make a list of positions), `tl.dot` (multiply two small matrices), and the *grid* (how many copies run). Our whole kernel is those five things.

**2. Your PMPP memory-chapter notes.** The one fact that rules everything: doing math is ~100× faster than fetching numbers from GPU memory. So a fast kernel fetches each number once and does lots of math on it. A slow kernel keeps going back to memory. That ratio — math per byte fetched — decides everything in this project.

**3. GB10 notes.** Your GPU's memory is slower than datacenter GPUs', so avoiding memory trips matters even *more* here. Absolute speeds won't transfer to big hardware; ratios will. (Same reporting rule as Project 1.)

**In the code:** open `flash_attn.py`, find `pid_m = tl.program_id(0)` — that's a copy of the program asking "which block am I?" And notice every `tl.load` has a `mask=` — that's how it safely handles edges that don't divide evenly.

---

## C2. Online softmax — the one trick — [CORE — week 1]

**Analogy:** you're finding the class average on a test, but exam papers arrive in batches, and the grading scale changes whenever a new highest score shows up. Instead of waiting for all papers, you keep two sticky notes: *highest score so far* and *running total (in the current scale)*. When a new batch has a higher max, you rescale your running total once, then add the new batch. Same final answer — but you never needed all the papers on your desk at once.

**1. The online-softmax paper.** Exactly that: softmax normally needs the whole row before it can start; the online version streams the row in chunks, carrying just those two sticky notes (`m` = max, `l` = sum). One rescale line does all the work.

**2. Why subtract the max at all:** `exp()` of a big number overflows. Subtracting the max first keeps everything small and safe. The online trick just maintains this safety continuously.

**In the code:** `online_softmax.py`, three lines — see the rescale:

```python
m_new = tl.maximum(m, tl.max(x, axis=0))
l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)   # rescale old, add new
m = m_new
```

---

## C3. Naive attention — feel the wall — [CORE — week 1]

**1. The attention formula (that's all you need from the famous paper).** Every word scores itself against every earlier word → a big table of scores (N×N) → scores become weights → weights average the values. That table is the villain.

**2. Why it's a wall.** At 8k context the score table has 67 *million* entries — per head, per layer. Writing it to slow memory and reading it back costs far more than the math itself. It's like doing your taxes but mailing every intermediate sum to a warehouse and waiting for it to come back before the next addition. The math is cheap; the shipping kills you.

**In the code:** `reference.py`'s `naive_attention()` builds the full table on purpose — one line, commented "the wall." We keep it forever: it's both the correctness answer key and the villain our kernel beats. The benchmark refuses to even run it past N=2048 — and reports that refusal as a result.

---

## C4. FlashAttention — never build the table — [CORE — weeks 1–2]

**The idea:** don't build the score table — build one *tile* of it at a time in the GPU's tiny fast memory, fold that tile into a running answer (using C2's sticky-note trick, plus a third note: the running output), throw the tile away, take the next one. The full table never exists anywhere. Memory shipping drops from "the whole table" to "each input read once." That's the entire famous paper.

**1. FlashAttention paper, sections 1–3** — the above, formally.
**2. FA2 paper, section 3** — the sequel is about *scheduling*: dividing the work across teams better. Read it to learn what production kernels beat you with (spoiler from our own gap analysis: this).
**3. Triton's fused-attention tutorial** — same kernel, tutorial style. Read *after* ours; it piles on advanced features that hide the plot.

**In the code:** `flash_attn.py`'s inner loop *is* the paper — seven lines: score one tile, update max, rescale old state, fold in new tile. Also spot: the **causal early-stop** (words can't see the future, so the loop just stops early — that's the whole 2× causal speedup), and the **GQA line** `h_kv = h // KV_GROUPS` (several query heads share one K/V head — one integer division implements the whole feature).

---

## C5. Prove it right before proving it fast — [CORE — week 2]

**1. BF16 basics.** Our numbers are stored in a format with only ~2–3 reliable decimal digits. So: always compare against a full-precision answer key (two sloppy answers can agree *and be wrong*), and use tolerances chosen for BF16, spelled out in the code — never defaults.

**2. Where kernel bugs hide:** at the edges. Sequence lengths that don't divide evenly by the block size, the diagonal of the causal mask, padding rows. A kernel tested only at nice round sizes is untested.

**In the code:** the tests deliberately use length **500** (the ugly one), both causal modes, and the real Qwen head ratios. And `assert_close()` doesn't just say "wrong" — it reports the worst-offending value and where it lives, so a failure is debuggable.

---

## C6. Benchmarking and the gap — [CORE — weeks 2–3]

**1. `do_bench` source (~50 lines).** The honest stopwatch: warmup first, many repetitions, robust average. Same discipline as Project 1. Never a single eyeballed run.

**2. The production rivals.** SGLang can serve attention through several engines (FlashInfer, its own Triton one, torch's built-in). On your unusual GPU, *which one actually runs* varies — so every result row records exactly which rival kernel executed. "X beat me" is meaningless without naming X's actual variant.

**3. ncu, three numbers only.** The profiler tells you *why* you lose: occupancy (does each GPU core have enough teams resident to stay busy while some wait for memory?), memory throughput vs peak (are you shipping-bound?), and what the workers are stalled on. Biggest difference vs the winner = the technique to borrow.

**4. The panel's rule:** a comparison table isn't the artifact. *Find why you lose, borrow one technique, measure the recovery* — that sentence is the write-up. (This already happened once: ncu said "not enough teams resident — register-starved," we restructured, and recovered half the FlashInfer gap. Read `p2/README.md`'s Result paragraph as the worked example.)

**In the code:** `bench_attn.py` — real model shapes only (14:2 heads, not 16:4), the FLOPs formula, and the dispatch record in every run's manifest.

---

## C7. Portability — [POST-SHIP]

Our kernel uses no NVIDIA-only instructions, so the same file compiles for AMD GPUs. One sentence in the write-up; a weekend MI300 cloud run would upgrade it from claimed to demonstrated.

---

## Sequencing

- **Week 1:** C1–C3 with the code open; run the test suite yourself in the container; then C4 slowly — it's the heart.
- **Week 2:** C5, then run `bench_attn` and read your own table.
- **Week 3:** the gap phase (C6) — profile, borrow, measure the delta. **Ship by Oct 1; fused RMSNorm is the fallback.**
- **Post-ship:** C7.

---

## Glossary — one line each

**Tile** — the small chunk of a matrix one program copy works on in fast memory. **Grid** — how many copies of the program launch. **Mask** — the "is this position real or past the edge?" check on loads/stores. **SRAM vs HBM** — tiny-fast on-chip memory (tiles live here) vs big-slow main memory (what FA avoids). **Occupancy** — how many teams each GPU core has resident; too few = the core naps while waiting on memory. **Registers** — each worker's pockets; run out and occupancy drops (our measured bottleneck). **Autotune** — try several block/team sizes, keep the fastest per shape. **num_stages** — how many memory fetches run ahead of the math, like preheating the next pan. **GQA** — several query heads share one K/V head; implemented by one integer divide. **Causal early-stop** — stop scoring at "now"; the future is invisible anyway. **BF16** — number format with big range, low precision; compare against FP32 with ~2e-2 tolerances. **do_bench** — the honest stopwatch. **Dispatch column** — record *which* rival kernel actually ran, or the comparison is unciteable.
