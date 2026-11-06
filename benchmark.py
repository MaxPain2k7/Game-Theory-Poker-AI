"""Compare CFR, CFR+ and DCFR on Kuhn and Leduc and plot exploitability.

    python benchmark.py --iters 10000

Writes results/convergence.png and results/benchmark.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poker_cfr import GameTree, CFRSolver, ALGOS, exploitability, expected_value

# colorblind-safe colors
COLORS = {"cfr": "#2a78d6", "cfr+": "#eb6834", "dcfr": "#1baf7a"}
LABELS = {"cfr": "CFR", "cfr+": "CFR+", "dcfr": "DCFR"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def run(game, algo, iters, checkpoints):
    tree = GameTree(game)
    solver = CFRSolver(tree, algo)
    xs, ys = [], []
    start = time.time()
    for c in checkpoints:
        solver.iterate(c - solver.iteration)
        xs.append(c)
        ys.append(exploitability(tree, solver.average_strategy()))
    avg = solver.average_strategy()
    return xs, ys, expected_value(tree, avg, avg), time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    args = ap.parse_args()

    checkpoints = np.unique(np.logspace(0, np.log10(args.iters), 40).astype(int))
    results = {}
    for game in ("kuhn", "leduc"):
        for algo in ALGOS:
            results[(game, algo)] = run(game, algo, args.iters, checkpoints)
            xs, ys, value, secs = results[(game, algo)]
            print(f"{game:<6} {algo:<5} exploitability {ys[-1]:.2e}  value {value:+.5f}  {secs:.1f}s")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, game, title in zip(axes, ("kuhn", "leduc"), ("Kuhn poker", "Leduc hold'em")):
        ax.set_facecolor(SURFACE)
        for algo in ALGOS:
            xs, ys, _, _ = results[(game, algo)]
            ax.loglog(xs, ys, color=COLORS[algo], lw=2, label=LABELS[algo])
        # end labels, nudged apart in log space so close lines don't collide
        ends = sorted(((np.log10(results[(game, a)][1][-1]), a) for a in ALGOS))
        placed = []
        for logy, algo in ends:
            if placed and logy - placed[-1] < 0.3:
                logy = placed[-1] + 0.3
            placed.append(logy)
            ax.annotate(LABELS[algo], (args.iters, 10 ** logy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("iterations")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, args.iters * 2.2)
    axes[0].set_ylabel("exploitability (chips per hand)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Average strategy exploitability, lower is closer to Nash", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    out = Path("results/convergence.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)

    lines = ["| game | algorithm | exploitability @ " + f"{args.iters:,}" + " iters | game value (seat 0) | time |",
             "|---|---|---|---|---|"]
    for (game, algo), (xs, ys, value, secs) in results.items():
        lines.append(f"| {game} | {LABELS[algo]} | {ys[-1]:.2e} | {value:+.5f} | {secs:.1f}s |")
    Path("results/benchmark.md").write_text("\n".join(lines) + "\n")
    print("wrote results/convergence.png and results/benchmark.md")


if __name__ == "__main__":
    main()
