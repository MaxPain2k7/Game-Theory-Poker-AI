"""Runs the Slumbot client against a local server that fails on purpose."""

import csv
from pathlib import Path

import pytest

import holdem_slumbot
from tests.mock_slumbot import MockSlumbot

BOT = Path(__file__).resolve().parent.parent / "models" / "holdem_bot.npz"
pytestmark = pytest.mark.skipif(not BOT.exists(), reason="needs the trained bot")


def run(mock, tmp_path, hands, *extra):
    return holdem_slumbot.main(["--host", mock.url, "--hands", str(hands), "--bot", str(BOT),
                                "--log-dir", str(tmp_path), "--max-wait", "0.01", "--timeout", "0.5",
                                "--delay", "0", *extra])


def logged(tmp_path):
    with (tmp_path / "slumbot_log.csv").open() as f:
        return [float(r["winnings"]) for r in csv.DictReader(f)]


def test_clean_server(tmp_path):
    with MockSlumbot(seed=1) as mock:
        assert run(mock, tmp_path, 60) == 0
    wins = logged(tmp_path)
    assert len(wins) == 60
    assert sorted(wins) == sorted(mock.delivered)       # every logged hand matches the server's record


def test_survives_every_kind_of_fault(tmp_path):
    faults = {"drop": 0.04, "lost_reply": 0.03, "error500": 0.03, "garbage": 0.03, "slow": 0.02, "expire": 0.02}
    with MockSlumbot(faults=faults, slow_seconds=1.0, seed=2) as mock:
        assert run(mock, tmp_path, 150) == 0
    wins = logged(tmp_path)
    assert len(wins) == 150
    assert all(v > 0 for v in mock.fault_counts.values()), mock.fault_counts
    # nothing the client logged was invented: every result is one the server really sent
    remaining = list(mock.delivered)
    for w in wins:
        remaining.remove(w)


def test_resume_keeps_old_hands(tmp_path):
    with MockSlumbot(seed=3) as mock:
        run(mock, tmp_path, 20)
        run(mock, tmp_path, 35, "--resume")
    assert len(logged(tmp_path)) == 35


def test_resume_ignores_half_written_line(tmp_path):
    (tmp_path / "slumbot_log.csv").write_text("hand,winnings,baseline_winnings,action,our_cards,board\n"
                                              "1,150,0,b300c,As Kd,\n2,")
    with MockSlumbot(seed=4) as mock:
        run(mock, tmp_path, 5, "--resume")
    assert len(logged(tmp_path)) >= 5


def test_gives_up_cleanly_when_server_is_gone(tmp_path):
    with MockSlumbot(seed=5) as mock:
        url = mock.url
    # the server is shut down now: every request fails to connect
    code = holdem_slumbot.main(["--host", url, "--hands", "10", "--bot", str(BOT), "--log-dir", str(tmp_path),
                                "--max-wait", "0.01", "--timeout", "0.3", "--delay", "0"])
    assert code == 0
    assert logged(tmp_path) == []


def test_search_bot_against_server(tmp_path):
    with MockSlumbot(seed=6) as mock:
        assert run(mock, tmp_path, 25, "--search") == 0
    assert sorted(logged(tmp_path)) == sorted(mock.delivered)
