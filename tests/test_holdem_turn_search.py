"""Turn search: values with the river dealt out, sensible strategies and the searching bot."""

from pathlib import Path

import numpy as np
import pytest

from poker_cfr.holdem.betting import Spot, STACK, FOLD
from poker_cfr.holdem.cards import parse_many
from poker_cfr.holdem.evaluator import eval7
from poker_cfr.holdem.search import COMBOS, combo_index, compatible_reach
from poker_cfr.holdem.turn_search import TurnSubgame

BOARD = parse_many("2c7d9hJs")
BOT = Path(__file__).resolve().parent.parent / "models" / "holdem_bot.npz"


def test_allin_runout_matches_brute_force():
    # seat 0 has shoved, seat 1 can fold or call and see the river
    spot = Spot(street=2, contrib=(STACK, 2000), street_in=(STACK - 2000, 0), to_act=1,
                last_raise=STACK - 2000, raises=1, acted=1)
    game = TurnSubgame(spot, BOARD)
    rng = np.random.default_rng(0)
    ranges = rng.random((2, len(COMBOS)))
    game.solve(ranges, iters=1, cards=48)
    game.reg_t[:] = 0                                   # uniform play: fold half, call half
    v = game.root_values(ranges, trav=1)
    opp = ranges[0] * game.valid_t
    deck = [c for c in range(52) if c not in BOARD]
    for hand in ("Td8d", "3c4c"):
        h = combo_index(*parse_many(hand))
        a, b = COMBOS[h]
        fold = -2000 * compatible_reach(COMBOS, opp, game.valid_t)[h]
        call = 0.0
        for o in np.nonzero(game.valid_t)[0]:
            c, d = COMBOS[o]
            if len({a, b, c, d}) < 4:
                continue
            rivers = [x for x in deck if x not in (a, b, c, d)]
            score = 0
            for x in rivers:
                board = np.array(BOARD + [x], np.int64)
                mine, theirs = eval7(a, b, board), eval7(c, d, board)
                score += (mine > theirs) - (mine < theirs)
            call += opp[o] * STACK * score / len(rivers)
        assert v[h] == pytest.approx(0.5 * fold + 0.5 * call)


def test_turn_strategy_is_sensible():
    spot = Spot(street=2, contrib=(600, 1200), street_in=(0, 600), to_act=0, last_raise=600, raises=1, acted=1)
    game = TurnSubgame(spot, BOARD).solve(np.ones((2, len(COMBOS))), iters=150, cards=8, seed=1)
    strat = game.root_strategy_all()
    kinds = [a[0] for a in game.root_actions()]
    nuts = combo_index(*parse_many("Td8d"))             # straight
    air = combo_index(*parse_many("4c3h"))              # nothing, no draw
    assert np.allclose(strat[:, game.valid_t].sum(0), 1)
    assert strat[kinds.index(FOLD), nuts] < 0.02
    assert strat[kinds.index(FOLD), air] > 0.5


@pytest.mark.skipif(not BOT.exists(), reason="needs the trained bot")
def test_turn_search_bot_plays():
    from poker_cfr.holdem.agents import CallingStation
    from poker_cfr.holdem.engine import play_hand
    from poker_cfr.holdem.turn_search import TurnSearchAgent
    bot = TurnSearchAgent.from_file(BOT, iters=40, turn_iters=20, turn_cards=4)
    rng = np.random.default_rng(3)
    for _ in range(8):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        assert play_hand(bot, CallingStation(), cards).over
    assert bot.turn_searches > 0
