"""Distribution-aware buckets: histograms, k-means and live bucketing."""

import numpy as np

from poker_cfr.holdem import abstraction
from poker_cfr.holdem.agents import CFRAgent, CallingStation
from poker_cfr.holdem.betting import BettingTree
from poker_cfr.holdem.cards import parse_many
from poker_cfr.holdem.engine import play_hand
from poker_cfr.holdem.evaluator import equity_histogram, equity_sampled


def hist(hand, board, runouts=400):
    np.random.seed(0)
    b = np.zeros(5, np.int64)
    cards = parse_many(board)
    b[:len(cards)] = cards
    h = parse_many(hand)
    return equity_histogram(h[0], h[1], b, len(cards), runouts, 60, 10), equity_sampled(h[0], h[1], b, len(cards), 4000)


def test_histogram_mean_matches_equity():
    h, eq = hist("QhJh", "Th9c2h")
    assert h.sum() == 400
    centers = (np.arange(10) + 0.5) / 10
    assert abs((h * centers).sum() / h.sum() - eq) < 0.05


def test_draw_and_made_hand_have_different_shapes():
    draw, _ = hist("AhKh", "7h2h9c")       # nut flush draw: mass near 0 and near 1
    pair, _ = hist("9d8s", "7h2h9c")       # top pair, weak kicker: mass in the middle
    spread = lambda h: (h[:3].sum() + h[-2:].sum()) / h.sum()
    assert spread(draw) > 0.5 > spread(pair)


def test_kmeans_finds_obvious_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0.2, 0.01, (500, 4))
    b = rng.normal(0.8, 0.01, (500, 4))
    c = abstraction.kmeans(np.vstack([a, b]).astype(np.float32), 2)
    assert np.allclose(np.sort(c[:, 0]), [0.2, 0.8], atol=0.02)


def test_dist_bot_plays():
    tree = BettingTree(buckets=8)
    rng = np.random.default_rng(1)
    policy = rng.random((tree.num_rows, 5)).astype(np.float32)
    cents = np.sort(rng.random((2, 8, abstraction.HIST_BINS)), axis=-1).astype(np.float32)
    river = np.linspace(0, 1, 9)[1:-1].astype(np.float32)
    bot = CFRAgent(tree, policy, None, kind="dist", centroids=cents, river_cuts=river)
    for _ in range(20):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        assert play_hand(bot, CallingStation(), cards).over
        assert play_hand(CallingStation(), bot, cards).over
