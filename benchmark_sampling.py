"""Full traversal CFR vs external sampling MCCFR, measured in nodes touched.

    python benchmark_sampling.py --budget 3e7

Wall time isn't a fair comparison here (one side is vectorized numpy, the
other walks one path at a time in Python), so both are charged for every
node they visit. Full CFR visits the whole tree once per player per
iteration. Writes results/sampling.png and results/sampling.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poker_cfr import GameTree, CFRSolver, exploitability
from poker_cfr.mccfr import ExternalSamplingMCCFR

COLORS = {"cfr": "#2a78d6", "cfr+": "#eb6834", "es-mccfr": "#1baf7a"}
LABELS = {"cfr": "CFR", "cfr+": "CFR+", "es-mccfr": "ES-MCCFR"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def run_full(tree, algo, budgets):
    solver = CFRSolver(tree, algo)
    per_iter = 2 * tree.num_nodes
    xs, ys = [], []
    for b in budgets:
        target = max(1, int(b // per_iter))
        if target <= solver.iteration:
            continue
        solver.iterate(target - solver.iteration)
        xs.append(solver.iteration * per_iter)
        ys.append(exploitability(tree, solver.average_strategy()))
    return xs, ys


def run_mccfr(tree, budgets, seed=0):
    m = ExternalSamplingMCCFR(tree, seed=seed)
    xs, ys = [], []
    for b in budgets:
        while m.nodes_touched < b:
            m.iterate(200)
        xs.append(m.nodes_touched)
        ys.append(exploitability(tree, m.average_strategy()))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3e7)
    args = ap.parse_args()
    budgets = np.logspace(4, np.log10(args.budget), 24)

    games = (("leduc", "Leduc hold'em (6 cards)"), ("leduc13", "Leduc-13 (26 cards)"))
    results = {}
    for game, _ in games:
        tree = GameTree(game)
        print(tree.summary())
        for algo in ("cfr", "cfr+"):
            t = time.time()
            results[(game, algo)] = run_full(tree, algo, budgets)
            print(f"  {algo:<9} {results[(game, algo)][1][-1]:.2e}  {time.time() - t:.0f}s")
        t = time.time()
        results[(game, "es-mccfr")] = run_mccfr(tree, budgets)
        print(f"  es-mccfr  {results[(game, 'es-mccfr')][1][-1]:.2e}  {time.time() - t:.0f}s")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (game, title) in zip(axes, games):
        ax.set_facecolor(SURFACE)
        ends = []
        for algo in ("cfr", "cfr+", "es-mccfr"):
            xs, ys = results[(game, algo)]
            ax.loglog(xs, ys, color=COLORS[algo], lw=2, label=LABELS[algo])
            ends.append((np.log10(ys[-1]), algo, xs[-1]))
        placed = []
        for logy, algo, x in sorted(ends):
            if placed and logy - placed[-1] < 0.3:
                logy = placed[-1] + 0.3
            placed.append(logy)
            ax.annotate(LABELS[algo], (x, 10 ** logy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("nodes touched")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(budgets[0] * 0.8, args.budget * 6)
    axes[0].set_ylabel("exploitability (chips per hand)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Same compute budget, full traversal vs sampling", x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/sampling.png", dpi=160, facecolor=SURFACE)

    marks = [1e5, 1e6, 1e7, args.budget]
    lines = ["| game | algorithm | " + " | ".join(f"{m:.0e} nodes" for m in marks) + " |",
             "|---|---|" + "---|" * len(marks)]
    for (game, algo), (xs, ys) in results.items():
        cells = []
        for m in marks:
            ok = [y for x, y in zip(xs, ys) if x <= m * 1.05]
            cells.append(f"{ok[-1]:.3f}" if ok else "not one iteration yet")
        lines.append(f"| {game} | {LABELS[algo]} | " + " | ".join(cells) + " |")
    Path("results/sampling.md").write_text("\n".join(lines) + "\n")
    print("wrote results/sampling.png and results/sampling.md")


if __name__ == "__main__":
    main()
