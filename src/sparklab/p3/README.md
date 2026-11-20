# P3: Tool-Loop Gym

Can the whole agentic-RL loop (environment, verifier, GRPO, SGLang rollouts) run on one DGX Spark, and how much does prefix caching save during training?

**Writeup:** [writeups/p3_tool_loop_gym.md](../../../writeups/p3_tool_loop_gym.md)

## Files

- [env.py](env.py): seeded key-value store with five tools; bad calls return error strings instead of raising
- [tasks.py](tasks.py): six task templates (difficulty 1–3), each with a state-based verifier and a known cheat that must score 0
- [eval_base.py](eval_base.py): base-model success rate per task, plus the rollout cache hit rate; writes `runs/p3/base_eval_<timestamp>.json`
- [miles_env/](miles_env/): the miles integration: rollout function, tool specs, dataset builder
- [tests/test_p3_tasks.py](../../../tests/test_p3_tasks.py): determinism, every cheat scores 0, every honest solution scores 1

## Running (on the Spark)

Base-model eval (SGLang server up, web UI stopped so it doesn't share the cache):

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash configs/spark_server.sh    # terminal 1
python -m sparklab.p3.eval_base --episodes 20                    # terminal 2
```

Training with miles (the docker command is at the top of the script):

```bash
SMOKE=1 bash /ws/scripts/run_gym_grpo.sh    # 2 steps, checks the loop works
SMOKE=0 bash /ws/scripts/run_gym_grpo.sh    # the full 50-step run
```

Then pull the reward curve and cache totals with `scripts/extract_grpo_stats.py`.
