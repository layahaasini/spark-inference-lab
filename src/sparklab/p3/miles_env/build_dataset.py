"""Emit the training/eval jsonl for miles from the calibrated task mix.

Each row: chat-format prompt (the goal; tool specs are injected by the chat
template at rollout time), and metadata carrying (task, seed) so the rollout
function can reconstruct the exact seeded environment. The mix defaults to the
2026-08-19 calibration: copy_value (20% base success, in-band) as the core,
increment_counter (70%) as the easy stabilizer, cleanup_threshold (5%) as
curriculum headroom.

  python -m sparklab.p3.miles_env.build_dataset --out data/gym_train.jsonl
"""

import argparse
import json
import pathlib

from ..tasks import TEMPLATES

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
        })
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    counts = {t: sum(1 for r in rows if r["metadata"]["task"] == t) for t in mix}
    print(f"wrote {len(rows)} rows to {out}  mix={counts}")


if __name__ == "__main__":
    main()
