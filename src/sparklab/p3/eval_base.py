"""Eval-before-you-train: measure each task's base success rate BEFORE any RL.

GRPO learns only from groups where some rollouts succeed and some fail. A
task the base model never solves (or always solves) gives zero advantage and
a flat curve. So each task's base success rate is measured first, and only
tasks in roughly the 10-60% band go into the training mix.

Every model call goes through the P1 client, so the prefix-cache hit rate of
the rollouts is recorded too.

Run (server up, UI stopped):
  python -m sparklab.p3.eval_base --episodes 20
"""

import argparse
import json
import pathlib

from ..bench import collect_manifest
from ..p1.client import DEFAULT_BASE_URL, generate, warmup
from .env import TOOL_DOCS, is_done, parse_tool_call
from .tasks import TEMPLATES

MAX_TURNS = 8

# Qwen ChatML markers. Instruct models need their chat template: the first
# eval sent raw text and scored 0% on every task (kept in runs/p3/). The
# transcript stays append-only: each turn adds one assistant block and one
# tool-result block at the end.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def run_episode(episode, base_url, max_new_tokens=192):
    """One rollout: the model proposes tool calls, the env runs them, and the
    transcript grows at the end. Returns (success, stats)."""
    initial = episode.env.snapshot()
    # fixed system block + task goal: the long shared prefix every turn resends
    ctx = (f"{IM_START}system\nYou are an agent operating a key-value store.\n"
           f"{TOOL_DOCS}{IM_END}\n"
           f"{IM_START}user\nTASK: {episode.goal}{IM_END}\n"
           f"{IM_START}assistant\n")
    prompt_toks = cached_toks = 0
    for _ in range(MAX_TURNS):
        r = generate(ctx, base_url=base_url, max_new_tokens=max_new_tokens, stop=[IM_END])
        prompt_toks += r.prompt_tokens
        cached_toks += r.cached_tokens
        call = parse_tool_call(r.text)
        if call is not None:
            name, args = call
            obs = episode.env.call(name, args)
            # append-only: close the assistant turn, add the tool result, reopen
            ctx += (f"{r.text.strip()}{IM_END}\n"
                    f"{IM_START}user\nTOOL_RESULT: {obs}{IM_END}\n"
                    f"{IM_START}assistant\n")
            continue
        if is_done(r.text):
            break
        ctx += (f"{r.text.strip()}{IM_END}\n"
                f"{IM_START}user\nTOOL_RESULT: ERROR: no tool call found; write exactly one "
                f"TOOL: {{...}} line, or write DONE if finished{IM_END}\n"
                f"{IM_START}assistant\n")
    return episode.verify(initial), {"prompt_tokens": prompt_toks, "cached_tokens": cached_toks,
                                     "env_steps": episode.env.steps}


def main():
    ap = argparse.ArgumentParser(description="Base success-rate eval across task templates")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--episodes", type=int, default=20, help="seeds per template")
    ap.add_argument("--tasks", nargs="*", default=sorted(TEMPLATES), choices=sorted(TEMPLATES))
    ap.add_argument("--out-dir", default="runs/p3")
    a = ap.parse_args()

    warmup(a.base_url)
    manifest = collect_manifest(extra={"args": vars(a)})
    results = []
    for name in a.tasks:
        wins, ptoks, ctoks = 0, 0, 0
        for seed in range(a.episodes):
            ep = TEMPLATES[name](seed)
            ok, stats = run_episode(ep, a.base_url)
            wins += ok
            ptoks += stats["prompt_tokens"]
            ctoks += stats["cached_tokens"]
        rate = wins / a.episodes
        band = "TRAINABLE" if 0.10 <= rate <= 0.60 else ("too hard" if rate < 0.10 else "too easy")
        results.append({"task": name, "difficulty": TEMPLATES[name](0).difficulty,
                        "success_rate": rate, "band": band,
                        "prompt_tokens": ptoks, "cached_tokens": ctoks,
                        "rollout_hit_rate": ctoks / max(ptoks, 1)})
        print(f"{name:20s} diff={results[-1]['difficulty']}  success={rate:5.1%}  [{band:9s}]  "
              f"rollout hit={results[-1]['rollout_hit_rate']:5.1%}")

    out = pathlib.Path(a.out_dir) / f"base_eval_{manifest['ts'].replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "results": results}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
