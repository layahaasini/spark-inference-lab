"""Make every figure in writeups/figs/ from the run files in runs/.

P1 runs are picked by the CLI args saved in each manifest (latest run per
workload shape), so no numbers are typed in by hand.

  python scripts/make_figs.py
"""

import glob
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sparklab.p1.tracegen import TraceConfig, gen_trace  # noqa: E402
FIGS = ROOT / "writeups" / "figs"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

POOL_TOKENS = 383_544          # server-printed capacity at MEM_FRAC=0.05
KNEE_PREDICTED = 73            # (POOL - 3,900) / 5,200, fixed before measuring

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def load_runs():
    """Latest P1 run per workload shape, keyed by the args each run recorded."""
    shapes = {}
    for path in sorted(glob.glob(str(ROOT / "runs" / "p1" / "pressure_*.json"))):
        data = json.loads(pathlib.Path(path).read_text())
        args = data["manifest"].get("args", {})
        if args.get("group_size", 1) > 1:
            key = "group"
        elif args.get("zipf_alpha", 0) > 0:
            key = "zipf"
        elif len(args.get("sessions", [])) >= 6:
            key = "knee"       # the small-pool even sweep
        else:
            key = "baseline"   # 2-32 sessions at the big pool (cache on or off)
        if key == "baseline" and all(r["hit_rate"] < 0.01 for r in data["results"]):
            key = "cacheoff"
        shapes[key] = data     # sorted() means later timestamps overwrite earlier
    return shapes


def attainable_hit(cfg):
    """Best possible hit rate for a trace under an infinite cache: each request
    can reuse at most its longest previously-seen prefix. Counted in words,
    which is fine for a ratio because words map to tokens evenly here."""
    events = gen_trace(cfg)
    head = None
    prev = {}
    total = cached = 0
    for ev in events:
        words = ev["prompt"].split()
        total += len(words)
        s = ev["session"]
        ref = prev.get(s) if s in prev else head
        if ref is not None:
            n = 0
            for a, b in zip(ref, words):
                if a != b:
                    break
                n += 1
            cached += n
        if head is None:
            head = words  # the first prompt stands in for the shared head
        prev[s] = words
    return cached / total


def cfg_for(run, n_sessions):
    a = run["manifest"]["args"]
    return TraceConfig(seed=a["seed"], n_sessions=n_sessions, turns=a["turns"],
                       sys_tokens=a["sys_tokens"], group_size=a.get("group_size", 1),
                       zipf_alpha=a.get("zipf_alpha", 0.0),
                       max_session_words=a.get("max_session_words"))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def knee_line(ax):
    ax.axvline(KNEE_PREDICTED, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(KNEE_PREDICTED, ax.get_ylim()[1] * 0.03, " predicted knee (73)",
            color=INK2, fontsize=9, ha="left", va="bottom")


def fig_hit_rate(shapes):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, color, label in (("knee", BLUE, "uniform"), ("zipf", ORANGE, "Zipf α=1.1")):
        rs = shapes[key]["results"]
        xs = [r["n_sessions"] for r in rs]
        ys = [r["hit_rate"] * 100 for r in rs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -3), color=INK2, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    knee_line(ax)
    ax.set_xlabel("concurrent sessions")
    ax.set_ylabel("radix-cache hit rate (%)")
    ax.set_title("Hit rate vs session pressure on a 383k-token pool", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "p1_hit_rate_vs_sessions.png", dpi=180)


def fig_hit_rate_normalized(shapes):
    """Measured hit rate as a fraction of each shape's own attainable maximum —
    the fair cross-shape comparison, since the raw ceilings differ by trace shape."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, color, label in (("knee", BLUE, "uniform"), ("zipf", ORANGE, "Zipf α=1.1")):
        rs = shapes[key]["results"]
        xs = [r["n_sessions"] for r in rs]
        ys = [r["hit_rate"] / attainable_hit(cfg_for(shapes[key], r["n_sessions"])) * 100 for r in rs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -3), color=INK2, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    knee_line(ax)
    ax.set_xlabel("concurrent sessions")
    ax.set_ylabel("share of attainable hits achieved (%)")
    ax.set_title("Normalized to each shape's own ceiling (infinite-cache maximum)", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "p1_hit_rate_normalized.png", dpi=180)


def fig_ttft(shapes):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rs = shapes["knee"]["results"]
    xs = [r["n_sessions"] for r in rs]
    for stat, color in (("p50", BLUE), ("p99", ORANGE)):
        ys = [r["ttft_ms"][stat] for r in rs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=stat)
        ax.annotate(stat, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -3), color=INK2, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    knee_line(ax)
    ax.set_xlabel("concurrent sessions")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Eviction lands tail-first: p99 rises across the knee, p50 barely moves", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "p1_ttft_tail_first.png", dpi=180)


def fig_tokens_avoided(shapes):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts = [16, 64, 96]
    series = (("knee", BLUE, "uniform"), ("zipf", ORANGE, "Zipf"), ("group", AQUA, "group ×4"))
    width = 0.26
    for i, (key, color, label) in enumerate(series):
        by_n = {r["n_sessions"]: r["tokens_avoided"] / 1e6 for r in shapes[key]["results"]}
        xs = [j + (i - 1) * width for j in range(len(counts))]
        ys = [by_n[n] for n in counts]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", color=INK2, fontsize=9)
    ax.set_xticks(range(len(counts)), [str(n) for n in counts])
    ax.set_xlabel("concurrent sessions")
    ax.set_ylabel("prefill tokens avoided (millions)")
    ax.set_title("Prefill tokens avoided by workload shape", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "p1_tokens_avoided.png", dpi=180)


def fig_p2_attention():
    """TFLOP/s at 8k context for each backend, from the final benchmark run."""
    path = sorted(glob.glob(str(ROOT / "runs" / "p2" / "attn_bench_*.json")))[-1]
    rows = [r for r in json.loads(pathlib.Path(path).read_text())["results"]
            if r["seq"] == 8192 and r["tflops"]]
    shapes = ["qwen2.5-0.5b", "qwen2.5-7b", "llama3-8b"]
    impls = (("triton-fa (ours)", BLUE, "ours (Triton)"), ("flashinfer", ORANGE, "FlashInfer"),
             ("torch-sdpa", AQUA, "torch SDPA"), ("sglang-triton", MUTED, "SGLang Triton"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.2
    for i, (impl, color, label) in enumerate(impls):
        by_shape = {r["shape"]: r["tflops"] for r in rows if r["impl"] == impl}
        xs = [j + (i - 1.5) * width for j in range(len(shapes))]
        ys = [by_shape.get(s, 0) for s in shapes]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", color=INK2, fontsize=8)
    ax.set_xticks(range(len(shapes)), shapes)
    ax.set_ylabel("TFLOP/s (causal, 8k tokens)")
    ax.set_title("Attention forward at 8k context on GB10", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left", ncol=4, fontsize=9)
    ax.set_ylim(0, 110)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_attention_8k.png", dpi=180)


def fig_p3_reward():
    """GRPO reward per rollout step, with a 5-step moving average."""
    stats = json.loads((ROOT / "runs" / "p3" / "grpo_gym_stats.json").read_text())
    xs = [r["rollout"] for r in stats["curve"]]
    ys = [r["raw_reward"] for r in stats["curve"]]
    avg = [sum(ys[max(0, i - 4):i + 1]) / len(ys[max(0, i - 4):i + 1]) for i in range(len(ys))]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, ys, color=BLUE, linewidth=1.2, marker="o", markersize=3.5, alpha=0.7,
            label="mean reward per step (64 episodes)")
    ax.plot(xs, avg, color=ORANGE, linewidth=2.2, label="5-step moving average")
    ax.axhline(ys[0], color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(xs[-1], ys[0], f"step 0 ({ys[0]:.2f})", color=INK2, fontsize=9, ha="right", va="bottom")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean episode reward")
    ax.set_title("GRPO on the tool-loop gym (Qwen2.5-1.5B, one GB10)", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "p3_reward_curve.png", dpi=180)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    shapes = load_runs()
    missing = {"knee", "zipf", "group"} - set(shapes)
    if missing:
        raise SystemExit(f"missing run shapes: {missing}")
    fig_hit_rate(shapes)
    fig_hit_rate_normalized(shapes)
    fig_ttft(shapes)
    fig_tokens_avoided(shapes)
    fig_p2_attention()
    fig_p3_reward()
    for f in sorted(FIGS.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
