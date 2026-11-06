"""River search: terminal values, solver convergence and the searching bot."""

from pathlib import Path

import numpy as np
import pytest

from poker_cfr.holdem.betting import Spot, FOLD
from poker_cfr.holdem.cards import parse_many
from poker_cfr.holdem.search import (COMBOS, RiverSubgame, showdown_values, compatible_reach,
                                     combo_index)

BOARD = parse_many("2c7d9hJsQh")
BOT = Path(__file__).resolve().parent.parent / "models" / "holdem_bot.npz"


@pytest.fixture(scope="module")
def game():
    spot = Spot(street=3, contrib=(1000, 1000), street_in=(0, 0), to_act=1, last_raise=100, raises=0, acted=0)
    return RiverSubgame(spot, BOARD)


def test_showdown_matches_brute_force(game):
    rng = np.random.default_rng(0)
    r = rng.random(len(COMBOS)) * game.valid
    fast = showdown_values(game.rank, COMBOS, r, game.order, 1.0)
    valid = np.nonzero(game.valid)[0]
    for h in valid[::37]:
        a, b = COMBOS[h]
        slow = sum(r[o] * np.sign(game.rank[h] - game.rank[o]) for o in valid
                   if len({a, b, *COMBOS[o]}) == 4)
        assert fast[h] == pytest.approx(slow)


def test_compatible_reach_excludes_blocked_hands(game):
    r = game.valid.astype(float)
    comp = compatible_reach(COMBOS, r, game.valid)
    # 47 unseen cards minus our 2: C(45, 2) opponent hands
    h = combo_index(*parse_many("AsKs"))
    assert comp[h] == pytest.approx(45 * 44 / 2)


def test_solver_converges(game):
    ranges = np.ones((2, len(COMBOS)))
    game.solve(ranges, iters=40)
    early = game.exploitability(ranges)
    game.solve(ranges, iters=300)
    late = game.exploitability(ranges)
    assert late < early / 3
    assert late < 0.01 * 2000             # under 1% of the pot


def test_nuts_never_fold_facing_a_bet():
    ranges = np.ones((2, len(COMBOS)))
    spot = Spot(street=3, contrib=(1000, 1750), street_in=(0, 750), to_act=0, last_raise=750, raises=1, acted=1)
    facing = RiverSubgame(spot, BOARD).solve(ranges, iters=300)
    nuts = combo_index(*parse_many("KcTc"))          # straight, the best hand here
    kinds = [a[0] for a in facing.nodes[0]["actions"]]
    p = dict(zip(kinds, facing.root_strategy_all()[:, nuts]))
    assert p.get(FOLD, 0) < 0.01


@pytest.mark.skipif(not BOT.exists(), reason="needs the trained bot")
def test_search_bot_plays_and_matches_blueprint_before_river():
    from poker_cfr.holdem.agents import CFRAgent, CallingStation
    from poker_cfr.holdem.engine import play_hand
    from poker_cfr.holdem.match import paired_comparison
    from poker_cfr.holdem.search import SearchAgent
    bp = CFRAgent.from_file(BOT)
    assert paired_comparison(CFRAgent.from_file(BOT), bp, CallingStation(), 20, seed=1).bb100 == 0.0
    sa = SearchAgent.from_file(BOT, iters=60)
    rng = np.random.default_rng(3)
    for _ in range(15):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        assert play_hand(sa, CallingStation(), cards).over
    assert sa.searches > 0
