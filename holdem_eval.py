"""Evaluate the trained hold'em bot.

    python holdem_eval.py
    python holdem_eval.py --bot models/holdem_bot_dist.npz --vs models/holdem_bot.npz

Duplicate matches against the baseline bots (and optionally another trained bot),
the bot's fallback rate (how often a real hand left its abstract tree) and a
chart of its small blind opening strategy. Writes results/holdem_eval*.md and
results/holdem_preflop*.png.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from poker_cfr.holdem.agents import CFRAgent, CallingStation, Maniac, RandomBot, EquityBot
from poker_cfr.holdem.cards import RANK_CHARS
from poker_cfr.holdem.match import duplicate_match

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def preflop_grid(bot):
    """13x13 share of hands the small blind raises (any size or all-in) at the start of the hand."""
    tree = bot.tree
    root = tree.find("")
    n = tree.n_actions[root]
    grid = np.zeros((13, 13))
    labels = [["" for _ in range(13)] for _ in range(13)]
    for i in range(13):          # row: first rank, high to low
        for j in range(13):
            hi, lo = 12 - min(i, j), 12 - max(i, j)
            if i == j:
                cls, lab = hi, RANK_CHARS[hi] * 2
            elif j > i:           # above diagonal: suited
                cls, lab = 13 + hi * (hi - 1) // 2 + lo, RANK_CHARS[hi] + RANK_CHARS[lo] + "s"
            else:
                cls, lab = 91 + hi * (hi - 1) // 2 + lo, RANK_CHARS[hi] + RANK_CHARS[lo] + "o"
            p = bot.policy[tree.row_base[root] + cls, :n].astype(float)
            p /= p.sum()
            grid[i, j] = 1 - p[0] - p[1]
            labels[i][j] = lab
    return grid, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="models/holdem_bot.npz")
    ap.add_argument("--deals", type=int, default=25000)
    ap.add_argument("--chart-only", action="store_true")
    ap.add_argument("--vs", help="another trained bot to play head to head, e.g. models/holdem_bot.npz")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    tag = Path(args.bot).stem.replace("holdem_bot", "")

    bot = CFRAgent.from_file(args.bot, seed=1)
    if args.chart_only:
        return draw_chart(bot, tag)
    meta = np.load(args.bot)
    lines = [f"Bot trained for {int(meta['iterations']):,} iterations, {int(meta['buckets'])} postflop buckets.", "",
             "| opponent | bot result | hands | bot left its tree |", "|---|---|---|---|"]
    opponents = [] if args.skip_baselines else [CallingStation(), Maniac(), RandomBot(3), EquityBot()]
    if args.vs:
        other = CFRAgent.from_file(args.vs, seed=2)
        other.name = f"CFR bot from {Path(args.vs).name}"
        opponents.append(other)
    for opp in opponents:
        bot.fallbacks = bot.decisions = 0
        t = time.time()
        res = duplicate_match(bot, opp, args.deals, seed=2026)
        lo, hi = res.ci95
        share = bot.fallbacks / max(bot.decisions, 1)
        print(f"{opp.name:16s} {res}  fallbacks {share:.1%}  ({time.time() - t:.0f}s)", flush=True)
        lines.append(f"| {opp.name} | {res.bb100:+.1f} bb/100 (95% CI {lo:+.1f} to {hi:+.1f}) | {res.hands:,} | {share:.1%} of decisions |")
    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/holdem_eval{tag}{'_vs' if args.vs else ''}.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    draw_chart(bot, tag)


def draw_chart(bot, tag=""):
    grid, labels = preflop_grid(bot)
    cmap = LinearSegmentedColormap.from_list("raise", ["#eef4fc", "#2a78d6", "#0d3a73"])
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    for i in range(13):
        for j in range(13):
            dark = grid[i, j] > 0.4
            ax.text(j, i - 0.15, labels[i][j], ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)
            ax.text(j, i + 0.22, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if dark else INK_MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("How often the bot raises first from the small blind\n(suited above the diagonal, offsuit below)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"results/holdem_preflop{tag}.png", dpi=160, facecolor=SURFACE)
    print(f"wrote results/holdem_preflop{tag}.png")


if __name__ == "__main__":
    main()
