"""Extract the publishable numbers from a GRPO run's artifacts.

Two sources, two outputs:
- the run log -> per-rollout reward/length series (the learning curve)
- the dump dir's rollout_data/<step>.pt -> per-sample radix accounting summed
  across the whole run (the P1<->P3 bridge number: prefill work the cache
  refunded DURING TRAINING)

Run inside the miles container (torch + miles importable):
  python3 /ws/scripts/extract_grpo_stats.py --log /ws/runs/grpo_train.log \
      --dumps /ws/runs/grpo_gym/dumps --out /ws/runs/grpo_gym_stats.json
"""

import argparse
import glob
import json
import os
import re


def parse_curve(log_path):
    rows = []
    pat = re.compile(r"rollout (\d+): \{(.*)")
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if not m or "raw_reward" not in line:
            continue
        def grab(key):
            k = re.search(rf"'{key}': ([-0-9.e]+)", m.group(2))
            return float(k.group(1)) if k else None
        rows.append({"rollout": int(m.group(1)), "raw_reward": grab("rollout/raw_reward"),
                     "response_len": grab("rollout/response_lengths"),
                     "truncated": grab("rollout/truncated")})
    # the log may contain a collapsed earlier run; keep the LAST monotonically
    # restarting sequence (rollout numbering resets to 0 on a new run)
    last_start = 0
    for i in range(1, len(rows)):
        if rows[i]["rollout"] < rows[i - 1]["rollout"]:
            last_start = i
    return rows[last_start:]


def sum_radix(dumps_dir):
    import torch
    total = {"prompt_tokens": 0, "cached_tokens": 0, "samples": 0, "env_steps": 0}
    for pt in sorted(glob.glob(os.path.join(dumps_dir, "rollout_data", "*.pt"))):
        data = torch.load(pt, map_location="cpu", weights_only=False)
        samples = data if isinstance(data, list) else data.get("samples", [])
        for s in samples:
            meta = getattr(s, "metadata", None) or (s.get("metadata") if isinstance(s, dict) else None) or {}
            radix = meta.get("radix") or {}
            total["prompt_tokens"] += radix.get("prompt_tokens", 0)
            total["cached_tokens"] += radix.get("cached_tokens", 0)
            total["env_steps"] += meta.get("env_steps", 0)
            total["samples"] += 1
    total["hit_rate"] = total["cached_tokens"] / max(total["prompt_tokens"], 1)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    curve = parse_curve(a.log)
    try:
        radix = sum_radix(a.dumps)
    except Exception as e:  # dumps format drift shouldn't lose the curve
        radix = {"error": f"{type(e).__name__}: {e}"}
    out = {"curve": curve, "radix": radix,
           "summary": {"rollouts": len(curve),
                       "first_reward": curve[0]["raw_reward"] if curve else None,
                       "last5_mean": (sum(r["raw_reward"] for r in curve[-5:]) / min(5, len(curve))) if curve else None}}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({**out["summary"], "radix": radix}, indent=2))


if __name__ == "__main__":
    main()
