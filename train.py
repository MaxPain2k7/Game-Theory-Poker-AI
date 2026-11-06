"""Train a strategy with CFR and save it.

    python train.py --game leduc --algo cfr+ --iters 10000
"""

import argparse
import csv
import time
from pathlib import Path

from poker_cfr import GameTree, CFRSolver, ALGOS, exploitability, expected_value
from poker_cfr import policy
from poker_cfr.games import GAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=GAMES, default="leduc")
    ap.add_argument("--algo", choices=ALGOS, default="cfr+")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=None, help="strategy json path")
    args = ap.parse_args()

    tree = GameTree(args.game)
    print(tree.summary())
    solver = CFRSolver(tree, args.algo)

    rows = []
    start = time.time()
    print(f"{'iter':>8} {'exploitability':>16} {'game value':>12} {'secs':>7}")
    while solver.iteration < args.iters:
        solver.iterate(min(args.eval_every, args.iters - solver.iteration))
        avg = solver.average_strategy()
        expl = exploitability(tree, avg)
        value = expected_value(tree, avg, avg)
        secs = time.time() - start
        rows.append((solver.iteration, expl, value, secs))
        print(f"{solver.iteration:>8} {expl:>16.8f} {value:>12.6f} {secs:>7.1f}")

    tag = f"{args.game}_{args.algo.replace('+', 'plus')}"
    out = Path(args.out or f"strategies/{tag}.json")
    policy.save(out, tree, solver.average_strategy(), algo=args.algo,
                iterations=solver.iteration, exploitability=rows[-1][1], game_value=rows[-1][2])
    log = Path(f"results/{tag}_log.csv")
    log.parent.mkdir(exist_ok=True)
    with log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "exploitability", "game_value_p0", "seconds"])
        w.writerows(rows)
    print(f"saved strategy to {out} and log to {log}")


if __name__ == "__main__":
    main()
