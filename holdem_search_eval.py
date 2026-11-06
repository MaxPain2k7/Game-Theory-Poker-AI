"""Measure what river search adds on top of the blueprint.

    python holdem_search_eval.py --opponent blueprint --deals 3000
    python holdem_search_eval.py --opponent equity --deals 3000

Paired comparison: every hand is played once by the plain blueprint bot and once
by the searching bot, with the same cards and the same random numbers. Both play
identically until the river, so the difference isolates what search changes.
Writes results/holdem_search_<opponent>.md.
"""

import argparse
import time
from pathlib import Path

from poker_cfr.holdem.agents import CFRAgent, EquityBot, CallingStation, Maniac
from poker_cfr.holdem.match import paired_comparison
from poker_cfr.holdem.search import SearchAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="models/holdem_bot.npz")
    ap.add_argument("--opponent", choices=["blueprint", "equity", "station", "maniac"], default="blueprint")
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base = CFRAgent.from_file(args.bot)
    new = SearchAgent.from_file(args.bot, iters=args.iters)
    villain = {"blueprint": lambda: CFRAgent.from_file(args.bot), "equity": EquityBot,
               "station": CallingStation, "maniac": Maniac}[args.opponent]()
    start = time.time()
    res = paired_comparison(new, base, villain, args.deals, seed=args.seed, progress=250)
    lo, hi = res.ci95
    secs = new.search_seconds / max(new.searches, 1)
    line = (f"search minus blueprint vs {args.opponent}: {res.bb100:+.1f} bb/100 (95% CI {lo:+.1f} to {hi:+.1f}), "
            f"{res.hands:,} paired hands, {new.searches:,} river searches at {secs:.2f}s each, "
            f"{(time.time() - start) / 60:.0f} min")
    print(line, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/holdem_search_{args.opponent}.md").write_text(line + "\n")


if __name__ == "__main__":
    main()
