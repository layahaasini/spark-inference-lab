# Project 3 Reading Companion — Plain and Simple

*Same format as the Project 1 and 2 companions: one paragraph per reading, plain language, definitions inline. Each section ends by pointing at the real code in this repo with a short excerpt. The gym, verifiers, and base-success eval are built and test-verified; the training run itself is the capstone sessions, gated on the miles integration. An equations-and-concepts glossary closes the document.*

---

## P L A N — A G E N T I C   R L   O N   O N E   S P A R K

## D1. The RL loop for language models — [CORE — week 1]

**1. Any PPO-for-LLMs explainer, skimmed for shape only.** The loop: the model (the *policy*) generates answers (*rollouts*), a scorer assigns each a number (*reward*), and an update nudges the model toward higher-scoring behavior. Everything else — advantages, clipping, KL penalties — is machinery to make that nudge stable. You need the shape, not the derivations.

**2. The GRPO section of the DeepSeekMath paper.** GRPO (Group Relative Policy Optimization) is the variant this project uses, and its core idea fits in a sentence: instead of training a second model to judge how good a reward is (PPO's critic), sample a *group* of n answers to the same prompt and score each one against the group's average. An answer's *advantage* is its reward minus the group mean, divided by the group's spread. Better-than-siblings gets pushed up, worse gets pushed down. No critic model, half the memory — the reason this fits on one box.

**3. The all-zero-group trap (Together's catch, chain doc).** GRPO's advantage is relative to the group. If every rollout in a group scores 0 — a task the model *never* solves — the advantage is 0 for all of them, the gradient is 0, and the curve is flat forever. Same if every rollout scores 1. The training signal lives strictly in *mixed* groups, which is why base success rates must sit in a middle band (roughly 10–60%) before training starts. This single fact drives the gym's difficulty ladder and the eval-first rule.

**The code, look for:**
- Nothing yet — this section is pure concept. The advantage arithmetic arrives with the miles recipe; the trap-avoidance machinery is D4's eval.

---

## D2. Tool-loop environments — [CORE — week 1]

**1. The ReAct paper, sections 1–2.** The agent pattern this gym models: the model alternates reasoning text with tool actions, each tool result is appended to the conversation, and the loop continues until done. Note what never happens: nothing above the append point is ever edited. That invariant is why agent traffic is cacheable — the P1 connection, now load-bearing for training economics.

**2. `poc_agentic_reward.py` (your own PoC on the Spark).** The predecessor this gym extends: one task, one verifier, the 0.5B model finding a lucky guess. Read it as the "before" picture — everything in D3 is that PoC's lessons institutionalized.

**3. Environment-design notes (Environments Hub docs, skimmed).** The venue conventions the deliverable cross-posts to: tasks as parameterized templates others can instantiate, seeded initial states, machine-checkable success. Reusability is what gets example PRs merged — the gym's structure is shaped by this from day one.

**The code, look for:**
- `ToolEnv`: state is a plain seeded dict; tools return observation *strings*; bad calls return error strings instead of raising — a confused model is an episode outcome, not a crash.
- `TOOL_DOCS` + the transcript assembly in `eval_base.run_episode`: fixed system+tools block, then append-only turns — P1's synthetic workload shape, now produced by a real loop.

```python
# src/sparklab/p3/env.py — errors are observations, never crashes:
if name == "write":
    if not isinstance(args["value"], int):
        return "ERROR: value must be an integer"
    self.state[args["key"]] = args["value"]
    return "OK"
```

---

## D3. Verifiers and reward hacking — [CORE — week 1]

**1. Any reward-hacking survey, one pass.** The optimizer will find every gap between "what the reward measures" and "what you meant." The classic agentic case: the checker greps the model's *output* for the right answer, so the model learns to *say* the answer without doing the task. Defense one: verify the environment's **state**, never the model's text.

**2. The port-8080 lesson (your PoC postmortem).** Your 0.5B model once scored reward by guessing a common constant instead of doing the work. The institutionalized fix: every task template ships its *own documented hack* — a `lucky_guess()` trajectory encoding the cheat a lazy policy would try — and a permanent test asserts it scores 0. A reward function whose hacks aren't tested is a reward function whose hacks will be found mid-training, expensively.

**3. The integrity idea (this repo's addition).** Surface-goal checks leave a subtler hack open: reach the goal by bulldozing everything else (delete all keys, then write the answer). So `verify()` here checks the goal *and* that unrelated state is untouched. Success = goal reached *and* nothing else changed.

**The code, look for:**
- `_unchanged_except()` — the integrity predicate, applied by every template.
- `cleanup_threshold`'s `lucky_guess` is literally the bulldozer: delete everything; the surface goal ("no key above threshold remains") holds, the verifier still says 0.
- `tests/test_p3_tasks.py::test_lucky_guess_scores_zero` — the permanent enforcement, per template, on every push.

```python
# src/sparklab/p3/tasks.py — the canonical hack, shipped with its task:
# the canonical hack: delete EVERYTHING — surface goal reached, integrity gone
_lucky=lambda e: [e.call("delete", {"key": k}) for k in list(e.state)],
```

---

## D4. The difficulty ladder and eval-before-train — [CORE — week 2]

**1. Curriculum notes (any RL curriculum explainer, skimmed).** Gradient signal must exist *somewhere* at every skill level: if all tasks are hard, groups are all-zero (D1's trap); if all are easy, all-one. The ladder here: difficulty 1 = one write, 2 = read-then-write, 3 = multi-step — so as the policy improves, some rung is always in the mixed-group band.

**2. The eval-first rule (chain doc, Together amendment).** Before any trainer exists: run each template against the base model N times, measure the success rate, and admit into the training mix only tasks in the ~10–60% band. A cheap afternoon pass that prevents the two-week flat-curve discovery. The band verdicts print as `TRAINABLE` / `too hard` / `too easy`.

**3. Model sizing (hardware panel, chain doc).** 1.5–4B with LoRA, never bigger: 128 GB unified memory *fits* 7–8B, but decode bandwidth (273 GB/s) is the rollout bottleneck, and group size × iteration speed matter more than model quality for a learning-curve deliverable. The 0.5B PoC sat at ~0% on tool tasks — the band rule made the sizing decision before any training run did.

**The code, look for:**
- `eval_base.py::main` — per-template success over seeds, band classification, one JSON per run with manifest (Project 1's discipline, unchanged).
- `test_difficulty_ladder_has_every_rung` — the ladder is a tested invariant, not an intention.

```python
# src/sparklab/p3/eval_base.py — the trainability band, mechanized:
band = "TRAINABLE" if 0.10 <= rate <= 0.60 else ("too hard" if rate < 0.10 else "too easy")
```

---

## D5. Rollouts and the caching bridge — [CORE — week 2]

**1. Your own P1 write-up, group-mode section.** GRPO's rollout shape — n continuations sharing one full context — is *exactly* the `group_size` mode your trace generator models, measured at ~96% hit rate. That number was the prediction; Project 3 measures it for real, inside training.

**2. SGLang rollout-serving notes (miles/verl docs on the rollout engine).** During training, rollouts are served by the same SGLang engine Project 1 measured — same radix cache, same economics. The co-headline deliverable (Baseten amendment): run P1's instrumentation *inside* the training run and state "prefix caching cut rollout prompt work by X%" as a measured number. The inference↔RL bridge, quantified.

**The code, look for:**
- `run_episode` sums `prompt_tokens` and `cached_tokens` from every model call — the measured client is the transport, so cache accounting rides along for free.
- `eval_base` already prints a per-task `rollout hit rate` — the bridge number's first appearance, before any trainer exists.

```python
# src/sparklab/p3/eval_base.py — P1's instruments inside the loop:
r = generate(ctx, base_url=base_url, max_new_tokens=max_new_tokens)
prompt_toks += r.prompt_tokens
cached_toks += r.cached_tokens
```

---

## D6. The miles recipe — [CORE — weeks 3–4, gated]

**1. miles recipe anatomy (existing GRPO recipe in the miles repo).** A recipe wires four things: the task/dataset source, the rollout engine config, the reward function, and the trainer hyperparameters. The deliverable is a *parameterized* recipe others can point at their own task templates — mergeable-by-design is what gets recipe PRs accepted.

**2. The two escape hatches (chain doc).** When the curve misbehaves, two files explain most failures: the logprob-comparison site (is the policy the trainer scores the same policy that generated the rollouts?) and the advantage-computation site (are groups being normalized as intended?). Two files, not the whole trainer — knowing where to look is the debugging skill this project trains.

**3. LoRA notes (any LoRA explainer, one pass).** Instead of updating all weights, train small low-rank additions to each weight matrix — a fraction of the memory and optimizer state. On unified memory it means model + rollout engine + trainer state coexist without discrete-VRAM juggling. `⚑ gated:` the actual training sessions start when the miles integration (#1714) lands.

**The code, look for:**
- Not in this repo yet by design — the recipe lands in the miles tree. The gym's contract (templates in, verified rewards out) is the stable interface it will consume.

---

## D7. Publishing curves honestly — [CONCURRENT with write-up]

**1. Reward-curve hygiene (any "how to read RL curves" note).** A learning curve without its evaluation protocol is decoration: state the eval task mix, the group size, how often eval runs, and the seed policy. Smoothing disclosed, never silent.

**2. Project 1's honesty section, re-read as a template.** The same moves transfer: state what the numbers are not, name the artifacts caught during the campaign, keep the run manifests. The rollout-economics claim gets the same treatment as the caching claims — per-run JSON, versions pinned, UI stopped during measurement windows.

---

## Sequencing (critical path)

- **Week 1 [CORE]:** D1–D3 readings against the working gym; run the test suite; write one new task template of your own (the fastest way to internalize the verifier rules).
- **Week 2 [CORE]:** D4–D5 — run `eval_base` on the Spark against 1.5B and 3B models; pick the training mix from the band table; record the baseline rollout hit rates.
- **Weeks 3–4 [CORE, gated on miles]:** D6 — recipe, small runs first, curves + rollout-economics instrumentation. **Escape hatches before heroics: logprob site, advantage site.**
- **[CONCURRENT]:** D7 — the write-up grows alongside the runs; mid-Nov artifact set: recipe PR + Environments Hub cross-post + curves + rollout-economics section.

---

## E Q U A T I O N S   &   K E Y   C O N C E P T S

**GRPO advantage.** For a group of n rollouts with rewards r₁…rₙ: advantage Aᵢ = (rᵢ − mean(r)) / std(r). All-equal rewards ⇒ std undefined/advantage 0 ⇒ no gradient — the mixed-group requirement in one formula.

**The trainability band.** Keep tasks with base success p in ~[0.1, 0.6]. Probability a group of n is mixed = 1 − pⁿ − (1−p)ⁿ; at p=0.02, n=8 that is ~0.15 — most groups teach nothing. At p=0.3 it is ~0.94.

**Rollout cache economics.** Per training step: prompt tokens sent × (1 − hit rate) = prefill actually computed. Group sampling reuses the shared context across n samples, so the attainable hit rate rises with n — P1's measured ~96% at n=4 is the predicted operating point.

**The concepts, one line each.** **Policy** — the model being trained. **Rollout** — one generated trajectory (here: a full tool-loop episode). **Reward** — the number the verifier assigns; here strictly {0, 1} from env state. **Group sampling** — n rollouts per prompt scored against each other; GRPO's replacement for a critic. **Advantage** — how much better than siblings; the actual training signal. **Verifier** — state predicate deciding reward; never reads model text. **Integrity check** — unrelated state must be untouched; closes the bulldozer hack. **Negative test** — a template's own documented hack, permanently asserted to score 0. **Difficulty ladder** — task tiers ensuring mixed groups exist at every skill level. **Base success rate** — pre-training success; band membership decides trainability. **LoRA** — low-rank adapters instead of full-weight updates; the memory trick that fits training on one box. **Escape hatches** — the logprob-comparison and advantage-computation sites; where misbehaving curves get diagnosed. **The bridge number** — prefill work avoided by the radix cache during training rollouts; Project 1 measured inside Project 3.
