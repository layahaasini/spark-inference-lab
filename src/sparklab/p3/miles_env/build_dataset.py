"""Emit the training/eval jsonl for miles from the calibrated task mix.

Each row has the goal as a chat prompt, the tool specs, and metadata with
(task, seed) so the rollout function can rebuild the same seeded env. The
default mix comes from the calibration run (runs/p3/base_eval_*14-18-49.json):
copy_value (20% base success) as the main task, increment_counter (70%) as an
easier one, cleanup_threshold (5%) as a harder one.

  python -m sparklab.p3.miles_env.build_dataset --out data/gym_train.jsonl
"""

import argparse
import json
import pathlib

from ..tasks import TEMPLATES
from .tools import TOOL_SPECS

DEFAULT_MIX = {"copy_value": 6, "increment_counter": 2, "cleanup_threshold": 2}  # per 10 rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gym_train.jsonl")
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=10_000,
                    help="offset so training seeds never overlap calibration seeds (0-19)")
    ap.add_argument("--mix", default=None, help='JSON like {"copy_value": 6, ...} weights per 10 rows')
    a = ap.parse_args()
    mix = json.loads(a.mix) if a.mix else DEFAULT_MIX
    assert all(t in TEMPLATES for t in mix), f"unknown task in mix; known: {sorted(TEMPLATES)}"

    cycle = [t for t, w in mix.items() for _ in range(w)]
    rows = []
    for i in range(a.rows):
        task = cycle[i % len(cycle)]
        seed = a.seed_base + i
        ep = TEMPLATES[task](seed)
        rows.append({
            "prompt": [{"role": "user", "content": ep.goal}],
            "label": "",                              # verifier is stateful; no text label
            "metadata": {"task": task, "seed": seed},
            # tools go in the row (--tool-key tools): miles applies the chat
            # template when it loads the dataset, and without tools there the
            # model writes prose instead of tool calls (all rewards were 0)
            "tools": TOOL_SPECS,
        })
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    counts = {t: sum(1 for r in rows if r["metadata"]["task"] == t) for t in mix}
    print(f"wrote {len(rows)} rows to {out}  mix={counts}")


if __name__ == "__main__":
    main()
