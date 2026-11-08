"""Measure how exploitable the hold'em bot is with Local Best Response.

    python holdem_lbr.py --blocks 20 --seed 1

Plays the bot against LBR with duplicate dealing (every deal four times, both
seats, both sets of hole cards), in blocks of 500 deals. Each finished block is
appended to results/holdem_lbr_blocks.csv straight away, so a run can stop at
any point and the pooled result still uses every finished block. LBR's
winnings are a lower bound on the bot's exploitability.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from poker_cfr.holdem.agents import CFRAgent
from poker_cfr.holdem.lbr import LBRAgent
from poker_cfr.holdem.match import duplicate_match

LOG = Path("results/holdem_lbr_blocks.csv")


def pooled(rows):
    means = np.array([float(r["bb100"]) for r in rows])
    ses = np.array([float(r["se"]) for r in rows])
    hands = sum(int(r["hands"]) for r in rows)
    mean = means.mean()
    se = np.sqrt((ses ** 2).sum()) / len(rows)
    return mean, se, hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="models/holdem_bot.npz")
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--deals-per-block", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    bot = CFRAgent.from_file(args.bot, seed=args.seed)
    lbr = LBRAgent(bot)
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["bot", "seed", "block", "hands", "bb100", "se"])
        for k in range(args.blocks):
            start = time.time()
            res = duplicate_match(lbr, bot, args.deals_per_block, seed=args.seed * 100_000 + k)
            w.writerow([Path(args.bot).name, args.seed, k, res.hands, round(res.bb100, 3), round(res.stderr, 3)])
            f.flush()
            with LOG.open(newline="") as g:
                rows = [r for r in csv.DictReader(g) if r["bot"] == Path(args.bot).name]
            mean, se, hands = pooled(rows)
            print(f"block {k}: {res.bb100:+.1f} bb/100 in {(time.time() - start) / 60:.0f} min | all blocks: "
                  f"LBR wins {mean:+.1f} bb/100 (95% CI {mean - 1.96 * se:+.1f} to {mean + 1.96 * se:+.1f}), "
                  f"{hands:,} hands", flush=True)


if __name__ == "__main__":
    main()
