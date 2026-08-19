# P3 — Tool-Loop Gym: end-to-end agentic RL on one box

**Question:** can the full agentic-RL loop — environments, verifiers, GRPO, SGLang rollouts — run on a single Spark, and what does prefix caching do to its economics?
**Method:** parameterized multi-turn tool-call task templates (seeded, machine-checkable, difficulty ladder 1→3), env-state verifiers that never trust model text, an anti-reward-hacking negative test per template, a base-success eval that gates the training mix to the 10–60% band, then GRPO via miles with SGLang rollouts — with P1's radix instrumentation running *inside* the training loop.
**Result (calibration measured 2026-08-19, Qwen2.5-1.5B):** the ladder spans the full range — write_constant 95%, increment_counter 70%, copy_value **20% (in the trainable band)**, cleanup_threshold 5%, largest_key/sum_two_keys 0% — so the training mix starts at copy_value with the harder rung-3 tasks as curriculum headroom. Real multi-turn rollouts run at **82–87% radix hit rate** (the P1↔P3 bridge, measured live). Calibration took three rounds, each catching a real instrument error before training could waste weeks on it: missing chat template (0% everywhere), then format-vs-competence confusion (models issuing correct calls in near-miss envelopes, scored 0). Training gate is OPEN: the rollout-rollback fix landed via miles #1826 (merged 2026-07-28; #1714 was its superseded twin). The training run is the next block of sessions.

**Study guide:** [docs/p3_reading_companion.md](../../../docs/p3_reading_companion.md)

## Pieces

- [env.py](env.py) — the tool environment: seeded key-value store, tools that return observation strings, errors as observations (never crashes), append-only transcripts (the P1 workload shape, for real)
- [tasks.py](tasks.py) — six templates across the difficulty ladder, each with a state-based verifier (integrity included: bulldozing unrelated state scores 0) and its own documented hack that must fail
- [eval_base.py](eval_base.py) — eval-before-you-train: base success per task via SGLang, band-membership verdicts, and per-rollout radix hit rates (the P1↔P3 bridge, measured before any trainer exists)
- [tests/test_p3_tasks.py](../../../tests/test_p3_tasks.py) — determinism, every hack scores 0, every honest solution scores 1, the ladder has every rung

## Running (on the Spark; server up, UI stopped)

```bash
python -m sparklab.p3.eval_base --episodes 20        # base success + trainability bands
```

## The rules this gym institutionalizes

- **Verify state, never text** — success is a predicate over the environment's final state; model claims are irrelevant.
- **Integrity is part of success** — reaching the goal while mutating unrelated state scores 0 (closes the delete-everything hack before the optimizer finds it).
- **Every template ships its own hack** — `lucky_guess()` must score 0, enforced by tests forever (the port-8080 lesson).
- **Eval before you train** — tasks outside the 10–60% base-success band produce all-zero or all-one groups: zero advantage, flat curves, weeks lost. Measure first.
- **Model sizing:** 1.5–4B + LoRA, never bigger — decode bandwidth is the rollout bottleneck on 273 GB/s, and group size × iteration speed beat model quality for a learning-curve deliverable.
- **The co-headline:** rollout cache economics measured during training — "prefix caching cut rollout cost by X%" is the inference↔RL bridge as a number.
