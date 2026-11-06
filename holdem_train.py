"""Train the no-limit hold'em bot.

    python holdem_train.py --minutes 90
    python holdem_train.py --minutes 90 --abstraction dist

Needs data/holdem_pool.npz (run holdem_build_pool.py first). Trains with
external sampling MCCFR for a time budget, checkpoints as it goes and can
resume. Writes models/holdem_ckpt.npz (full regrets, for resuming) and
models/holdem_bot.npz (just the playing strategy) plus a training log.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from poker_cfr.holdem import abstraction
from poker_cfr.holdem.agents import CFRAgent, EquityBot
from poker_cfr.holdem.betting import BettingTree
from poker_cfr.holdem.cards import parse_many, preflop_class
from poker_cfr.holdem.match import duplicate_match
from poker_cfr.holdem.trainer import HoldemTrainer


def save_bot(trainer, path, card_info):
    np.savez_compressed(path, policy=trainer.policy().astype(np.float16),
                        buckets=trainer.tree.buckets, iterations=trainer.iterations, **card_info)


def card_abstraction(pool, kind, k):
    """(buckets per deal, info saved with the bot so it can bucket live hands)."""
    if kind == "equity":
        cuts = abstraction.thresholds(pool["equity"], k)
        return abstraction.bucketize(pool, cuts), {"kind": "equity", "cuts": cuts}
    cache = Path(f"data/holdem_dist_buckets_{k}.npz")
    if cache.exists():
        d = np.load(cache)
        buckets, cents, river_cuts = d["buckets"], d["centroids"], d["river_cuts"]
    else:
        hists = np.load("data/holdem_hist.npz")["hists"]
        buckets, cents, river_cuts = abstraction.distribution_buckets(pool, hists, k)
        np.savez_compressed(cache, buckets=buckets, centroids=cents, river_cuts=river_cuts)
    info = {"kind": "dist", "centroids": cents, "river_cuts": river_cuts,
            "cuts": np.zeros((3, k - 1), np.float32)}
    return buckets, info


def preflop_summary(trainer):
    tree = trainer.tree
    root = tree.find("")
    pol = trainer.policy()
    out = []
    for hand in ("AsAh", "KsQs", "7c2d", "9h8h"):
        row = tree.row_base[root] + preflop_class(*parse_many(hand))
        p = pol[row, :tree.n_actions[root]]
        label = hand[0] * 2 if hand[0] == hand[2] else f"{hand[0]}{hand[2]}{'s' if hand[1] == hand[3] else 'o'}"
        out.append(f"{label} fold {p[0]:.2f} raise {1 - p[0] - p[1]:.2f}")
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--buckets", type=int, default=50)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--discount-share", type=float, default=0.3, help="share of the budget with linear discounting")
    ap.add_argument("--prune-after", type=float, default=0.2, help="start pruning after this share of the budget")
    ap.add_argument("--checkpoint-minutes", type=float, default=15)
    ap.add_argument("--eval-deals", type=int, default=300)
    ap.add_argument("--pool", default="data/holdem_pool.npz")
    ap.add_argument("--abstraction", choices=["equity", "dist"], default="equity",
                    help="equity: buckets by equity. dist: flop/turn buckets by equity histogram clusters "
                         "(needs data/holdem_hist.npz from holdem_build_hist.py)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pool = abstraction.load(args.pool)
    buckets, card_info = card_abstraction(pool, args.abstraction, args.buckets)
    tree = BettingTree(buckets=args.buckets)
    print(tree.summary(), flush=True)
    print(f"deal pool: {len(pool['winner']):,} deals", flush=True)

    trainer = HoldemTrainer(tree, buckets, pool["winner"])
    Path("models").mkdir(exist_ok=True)
    tag = "" if args.abstraction == "equity" else "_dist"
    ckpt, bot_path = f"models/holdem_ckpt{tag}.npz", f"models/holdem_bot{tag}.npz"
    if args.resume and Path(ckpt).exists():
        trainer.load(ckpt)
        print(f"resumed at {trainer.iterations:,} iterations", flush=True)

    log_path = Path(f"results/holdem_train_log{tag}.csv")
    log_path.parent.mkdir(exist_ok=True)
    new_log = not (args.resume and log_path.exists())
    log = log_path.open("w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["minutes", "iterations", "iters_per_sec", "vs_equity_bot_bb100", "ci95"])

    budget = args.minutes * 60
    start = last_ckpt = time.time()
    elapsed = 0.0
    while elapsed < budget:
        share = elapsed / budget
        t = time.time()
        trainer.run_batch(args.batch, prune=share >= args.prune_after, discount=share < args.discount_share)
        rate = args.batch / (time.time() - t)
        elapsed = time.time() - start
        if time.time() - last_ckpt >= args.checkpoint_minutes * 60 or elapsed >= budget:
            last_ckpt = time.time()
            trainer.save(ckpt, card_info["cuts"])
            save_bot(trainer, bot_path, card_info)
            bot = CFRAgent.from_file(bot_path)
            res = duplicate_match(bot, EquityBot(), args.eval_deals, seed=123)
            writer.writerow([round(elapsed / 60, 1), trainer.iterations, int(rate), round(res.bb100, 1),
                             round(1.96 * res.stderr, 1)])
            log.flush()
            print(f"[{elapsed / 60:6.1f} min] {trainer.iterations:>13,} iters  {rate:>7,.0f} it/s  "
                  f"vs equity bot {res}", flush=True)
            print(f"           {preflop_summary(trainer)}", flush=True)
    log.close()
    print(f"done: {trainer.iterations:,} iterations, saved {bot_path}", flush=True)


if __name__ == "__main__":
    main()
