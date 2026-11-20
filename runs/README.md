# runs/

Raw results. Every JSON has a `manifest` block (versions, GPU, server
settings, CLI args, seeds) next to its `results`, so each number in the
writeups can be traced back to the run that produced it.

## p1/ — prefix caching

| file | what it is |
|---|---|
| `pressure_2026-08-13T21-43-59.json` | first baseline sweep, 2–32 sessions (before the warmup fix) |
| `pressure_2026-08-14T10-53-32.json` | first knee sweep on the small pool, 16–96 sessions (before the warmup fix) |
| `pressure_2026-08-14T11-09-47.json` | cache-off baseline (before the warmup fix) |
| `pressure_2026-08-14T11-16-11.json` | knee sweep, 16–160 sessions, uniform scheduling |
| `pressure_2026-08-14T11-24-42.json` | knee sweep, Zipf α=1.1, 5,500-word session cap |
| `pressure_2026-08-14T11-30-25.json` | group mode (4 samples per context) |
| `pressure_2026-08-14T11-44-26.json` | **baseline sweep used in the writeup** (cache on, big pool) |
| `pressure_2026-08-14T11-45-43.json` | **cache-off baseline used in the writeup** |
| `pressure_2026-08-14T11-47-30.json` | **knee sweep used in the writeup** (uniform, rerun) |
| `pressure_2026-08-14T13-32-20.json`, `pressure_2026-08-15T13-45-42.json` | short re-checks of the baseline after ops changes |
| `sweep_*.log` | console output of the first sweeps (note the ~3 s p99 on the first point: the warmup bug) |
| `rerun_*.log` | console output of the reruns after the warmup fix, incl. the cold/warm check |

## p2/ — attention kernels

| file | what it is |
|---|---|
| `attn_bench_2026-08-19T19-58-31.json` | first benchmark: ours vs naive vs torch SDPA |
| `attn_bench_2026-08-19T20-07-57.json` | adds FlashInfer |
| `attn_bench_2026-08-19T20-13-48.json`, `…20-16-08.json` | tuning rounds 1 and 2 (block sizes / warps / stages) |
| `attn_bench_2026-08-19T20-28-37.json` | after the split loop + base-2 softmax change |
| `attn_bench_2026-08-19T20-40-39.json` | **final table used in the writeup** (adds SGLang's Triton backend) |
| `bench_full.log` | console output of the final run |
| `ncu_fa_8k.txt` | Nsight Compute report for our kernel at 8k (the occupancy diagnosis) |

## p3/ — tool-loop gym

| file | what it is |
|---|---|
| `base_eval_2026-08-19T13-49-32.json` | calibration round 1: raw text, no chat template (0% everywhere) |
| `base_eval_2026-08-19T14-05-12.json` | calibration round 2: chat template, strict call parser |
| `base_eval_2026-08-19T14-18-49.json` | **round 3, used in the writeup**: chat template + worked example in the tool docs + lenient call parser |
| `eval_base_1.5b*.log` | console output of rounds 1 and 3 |
| `grpo_train.log` | full log of the 50-step training run (lr=1e-6). The earlier lr=1e-5 attempt that collapsed is not included |
| `grpo_gym_stats.json` | reward curve + cache totals pulled out of the log and rollout dumps by `scripts/extract_grpo_stats.py` |
