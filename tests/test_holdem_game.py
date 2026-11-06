"""Rules, betting tree, action translation and the Slumbot parser."""

import numpy as np
import pytest

from poker_cfr.holdem.betting import BettingTree, STACK, FOLD, CALL, RAISE, ALLIN
from poker_cfr.holdem.cards import parse_many
from poker_cfr.holdem.engine import HandState, play_hand
from poker_cfr.holdem.agents import CFRAgent, CallingStation, Maniac, RandomBot


@pytest.fixture(scope="module")
def tree():
    return BettingTree(buckets=8)


def new_state():
    c = parse_many("AsAh") + parse_many("KdKc") + parse_many("2c7d9hJsQh")
    return HandState(c[0:2], c[2:4], c[4:9])


def run(tokens):
    s = new_state()
    for t in tokens:
        s.apply(t)
    return s


def test_limp_gives_big_blind_the_option():
    s = run(["c"])
    assert not s.over and s.street == 0 and s.to_act == 1
    s.apply("k")
    assert s.street == 1 and s.to_act == 1 and s.pot == 200


def test_min_raise_rules():
    s = run(["b300"])              # raise by 200
    assert s.raise_bounds() == (500, 20000)
    assert not s.legal("b400")
    s.apply("b1000")               # reraise by 700
    assert s.raise_bounds() == (1700, 20000)


def test_slumbot_example_history():
    s = new_state()
    for t in ["c", "b300", "c", "k", "k", "k", "b300", "c", "k", "b1200", "c"]:
        s.apply(t)
    assert s.over and s.history == "cb300c/kk/kb300c/kb1200c"
    assert s.payoff() == 1800      # aces hold, everyone put in 1,800


def test_allin_runs_out_the_board():
    s = run(["b20000", "c"])
    assert s.over and s.folded is None and s.payoff() == STACK


def test_tree_matches_engine_rules(tree):
    """Every abstract action must be legal in the real engine when replayed."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = new_state()
        node = 0
        while not tree.is_terminal[node]:
            n = tree.n_actions[node]
            a = int(rng.integers(n))
            kind, to = tree.kind[node, a], tree.raise_to[node, a]
            token = {FOLD: "f", CALL: "c" if s.to_call() else "k"}.get(kind, f"b{to}")
            assert s.legal(token), (tree.histories[node], token)
            assert s.to_act == tree.player[node] and s.street == tree.street[node]
            s.apply(token)
            node = tree.child[node, a]
        assert s.over
        assert list(tree.contrib[node]) == s.contrib


def test_translation_maps_onto_neighbours(tree):
    policy = np.ones((tree.num_rows, 5), np.float32)
    cuts = np.tile(np.linspace(0, 1, 9)[1:-1], (3, 1)).astype(np.float32)
    bot = CFRAgent(tree, policy, cuts, samples=50)
    s = new_state()
    root = tree.find("")
    kinds = list(tree.kind[root, :tree.n_actions[root]])
    picks = set()
    for _ in range(200):
        bot.new_hand()
        picks.add(bot._translate(root, s, "b330"))     # between the 2.5bb and 4bb opens
    assert picks <= {kinds.index(RAISE), kinds.index(RAISE) + 1} and len(picks) == 2
    assert bot._translate(root, s, "b15000") == kinds.index(ALLIN)


def test_bot_plays_full_hands_against_anything(tree):
    policy = np.random.default_rng(1).random((tree.num_rows, 5)).astype(np.float32)
    cuts = np.tile(np.linspace(0, 1, 9)[1:-1], (3, 1)).astype(np.float32)
    bot = CFRAgent(tree, policy, cuts, samples=50)
    rng = np.random.default_rng(2)
    for opp in (CallingStation(), Maniac(), RandomBot(4)):
        for _ in range(30):
            cards = [int(c) for c in rng.permutation(52)[:9]]
            assert play_hand(bot, opp, cards).over
            assert play_hand(opp, bot, cards).over


def test_slumbot_parser():
    from holdem_slumbot import rebuild
    resp = {"action": "b300c/kb200", "hole_cards": ["Ac", "Kd"], "board": ["2c", "7d", "9h"], "client_pos": 0}
    s = rebuild(resp, seat=1)
    assert s.street == 1 and s.to_act == 1 and s.to_call() == 200
    assert s.board == tuple(parse_many("2c7d9h"))


def test_trainer_learns_basic_preflop():
    from poker_cfr.holdem import abstraction
    from poker_cfr.holdem.trainer import HoldemTrainer
    from poker_cfr.holdem.cards import preflop_class
    pool = abstraction.build_pool(20000, 50, seed=3, log=None)
    cuts = abstraction.thresholds(pool["equity"], 8)
    tree = BettingTree(buckets=8)
    t = HoldemTrainer(tree, abstraction.bucketize(pool, cuts), pool["winner"], seed=1)
    for _ in range(6):
        t.run_batch(50_000, discount=True)
    pol = t.policy()
    assert np.allclose(pol.sum(axis=1), 1, atol=1e-3)
    root = tree.find("")
    aces = pol[tree.row_base[root] + preflop_class(*parse_many("AsAh")), 0]
    assert aces < 0.05      # never folds aces first in
