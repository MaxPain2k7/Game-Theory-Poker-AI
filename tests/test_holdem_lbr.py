"""Local Best Response: equity against a range, range tracking and full hands."""

from pathlib import Path

import numpy as np
import pytest

from poker_cfr.holdem.cards import parse_many
from poker_cfr.holdem.evaluator import equity_sampled
from poker_cfr.holdem.lbr import showdown_equity, LBRAgent
from poker_cfr.holdem.search import COMBOS, combo_index, valid_mask

BOT = Path(__file__).resolve().parent.parent / "models" / "holdem_bot.npz"


def uniform_range(hole, board):
    w = valid_mask(board).astype(float)
    for c in hole:
        w[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)] = 0
    return w / w.sum()


@pytest.mark.parametrize("hand,board", [("AhKh", "7h2h9c"), ("9d8s", "7h2h9cKd"), ("QsQc", "2c7d9hJsKh")])
def test_equity_against_uniform_range(hand, board):
    h, bd = parse_many(hand), parse_many(board)
    b = np.zeros(5, np.int64)
    b[:len(bd)] = bd
    exact = showdown_equity(h[0], h[1], b, len(bd), COMBOS, uniform_range(h, bd))
    np.random.seed(0)
    assert exact == pytest.approx(equity_sampled(h[0], h[1], b, len(bd), 100000), abs=0.01)


def test_equity_against_a_single_hand():
    h, bd = parse_many("AsAd"), parse_many("2c7d9hJsKh")
    w = np.zeros(len(COMBOS))
    w[combo_index(*parse_many("KcKd"))] = 1.0       # a set of kings beats the aces
    b = np.array(bd, np.int64)
    assert showdown_equity(h[0], h[1], b, 5, COMBOS, w) == 0.0


@pytest.mark.skipif(not BOT.exists(), reason="needs the trained bot")
def test_lbr_tracks_range_and_plays():
    from poker_cfr.holdem.agents import CFRAgent
    from poker_cfr.holdem.engine import play_hand
    bot = CFRAgent.from_file(BOT, seed=1)
    lbr = LBRAgent(bot)
    rng = np.random.default_rng(5)
    for _ in range(10):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        state = play_hand(lbr, bot, cards)
        assert state.over
        if lbr.range is not None:
            assert np.all(np.isfinite(lbr.range)) and lbr.range.min() >= 0
            for c in state.hole[0]:                     # LBR's own cards are never in the bot's range
                assert lbr.range[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)].sum() == 0
