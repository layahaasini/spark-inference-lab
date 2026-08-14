"""Generate the write-up figures from committed run JSONs.

Selects runs by their recorded CLI args (latest timestamp per shape), so the
figures always reflect the newest data in runs/ — never hand-entered numbers.
Output: writeups/figs/*.png
"""

import glob
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "writeups" / "figs"

# palette: categorical slots 1-3, chrome/ink roles
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
    """Latest run per shape, keyed by the args each run recorded."""
    shapes = {}
    for path in sorted(glob.glob(str(ROOT / "runs" / "pressure_*.json"))):
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
    fig.savefig(FIGS / "hit_rate_vs_sessions.png", dpi=180)


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
    fig.savefig(FIGS / "ttft_tail_first.png", dpi=180)


def fig_tokens_avoided(shapes):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts = [16, 64, 96]
    series = (("knee", BLUE, "uniform"), ("zipf", ORANGE, "Zipf"), ("group", AQUA, "group ×4"))
    width = 0.26
    for i, (key, color, label) in enumerate(series):
        by_n = {r["n_sessions"]: r["tokens_avoided"] / 1e6 for r in shapes[key]["results"]}
        xs = [j + (i - 1) * width for j in range(len(counts))]
        ys = [by_n[n] for n in counts]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label)  # slight shrink = surface gap
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", color=INK2, fontsize=9)
    ax.set_xticks(range(len(counts)), [str(n) for n in counts])
    ax.set_xlabel("concurrent sessions")
    ax.set_ylabel("prefill tokens avoided (millions)")
    ax.set_title("The headline metric by workload shape", loc="left")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "tokens_avoided.png", dpi=180)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    shapes = load_runs()
    missing = {"knee", "zipf", "group"} - set(shapes)
    if missing:
        raise SystemExit(f"missing run shapes: {missing}")
    fig_hit_rate(shapes)
    fig_ttft(shapes)
    fig_tokens_avoided(shapes)
    for f in sorted(FIGS.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
