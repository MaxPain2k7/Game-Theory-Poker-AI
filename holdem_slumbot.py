"""Play the trained bot against Slumbot (slumbot.com) through its public API.

    python holdem_slumbot.py --hands 5000
    python holdem_slumbot.py --hands 5000 --resume          # carry on after a stop
    python holdem_slumbot.py --hands 5000 --search --log-dir results/search   # with river search
    python holdem_slumbot.py --hands 5000 --turn-search --log-dir results/turn   # turn and river search
    python holdem_slumbot.py --hands 5000 --username you --password secret

Run this on your own machine, it needs internet access. Slumbot plays heads up
no-limit with 20,000 chip stacks and 50/100 blinds, the same format the bot
was trained for. Every finished hand is appended to results/slumbot_log.csv
straight away, so a crash or Ctrl+C never loses finished hands. Anything that
goes wrong is written to results/slumbot_errors.log.

Protocol: POST /api/new_hand {token} starts a hand, POST /api/act {token, incr}
sends an action. Replies carry the action history (same format as engine.py),
our hole cards, the board, our seat (client_pos 0 = big blind) and the result
when the hand ends.

Connection failures are retried with backoff. A move whose reply is lost is
never resent, since the server may already have applied it: that hand is
dropped and not counted. Expired sessions are restarted, rejected or
unparseable hands are logged and skipped. 25 failures in a row stop the run.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_HOST = "https://slumbot.com"
MAX_ABORTS_IN_A_ROW = 25


class HandAborted(Exception):
    """This hand can't be finished cleanly. It is skipped and not counted."""


class SessionLost(Exception):
    """The server no longer recognises our session token."""


class Client:
    def __init__(self, host, timeout=30.0, retries=6, max_wait=60.0, log=print):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.max_wait = max_wait
        self.log = log
        self.network_retries = 0

    def _once(self, path, body):
        req = urllib.request.Request(self.host + path, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "poker-cfr-client"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
            try:
                data = json.loads(raw)
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("error_msg"):
                return data                     # a real answer that happens to be an error
            if e.code in (429,) or e.code >= 500:
                raise ConnectionError(f"HTTP {e.code}")
            raise HandAborted(f"HTTP {e.code}: {raw[:200]}")
        try:
            data = json.loads(raw)
        except ValueError:
            raise ConnectionError(f"reply was not JSON: {raw[:80]!r}")
        if not isinstance(data, dict):
            raise ConnectionError(f"reply was not a JSON object: {raw[:80]!r}")
        return data

    def post(self, path, payload, idempotent=True):
        """Send a request, retrying network trouble.

        A failure while connecting means the server never saw the request, so it is
        always safe to retry. A failure after the request went out (connection reset,
        read timeout, a 5xx, a garbled reply) might mean the server already applied it.
        For /api/act that would risk sending the same move twice, so the hand is
        abandoned instead. Starting a new hand is safe to repeat either way."""
        body = json.dumps(payload).encode()
        for attempt in range(self.retries):
            try:
                data = self._once(path, body)
            except urllib.error.URLError as e:          # never reached the server
                self._pause(attempt, getattr(e, "reason", e))
                continue
            except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
                if not idempotent:
                    raise HandAborted(f"reply lost after sending a move ({e.__class__.__name__}: {e})")
                self._pause(attempt, e)
                continue
            msg = data.get("error_msg")
            if msg:
                low = str(msg).lower()
                if "token" in low or "session" in low or "expired" in low or "login" in low:
                    raise SessionLost(msg)
                raise HandAborted(f"server said {msg!r}")
            return data
        raise HandAborted(f"no usable reply from {path} after {self.retries} tries")

    def _pause(self, attempt, reason):
        self.network_retries += 1
        wait = min(self.max_wait, 2.0 * 2 ** attempt)
        self.log(f"  network problem ({reason}), retrying in {wait:.0f}s")
        time.sleep(wait)


def rebuild(resp, seat):
    """Our engine state from Slumbot's reply. The opponent's hidden cards and the
    undealt board are filled with placeholders the bot never looks at."""
    from poker_cfr.holdem.cards import parse_many
    from poker_cfr.holdem.engine import HandState

    mine = parse_many(resp["hole_cards"])
    board = parse_many(resp.get("board") or [])
    used = set(mine) | set(board)
    if len(mine) != 2 or len(used) != len(mine) + len(board) or len(board) > 5:
        raise ValueError(f"bad cards in reply: {resp.get('hole_cards')} {resp.get('board')}")
    filler = [c for c in range(52) if c not in used]
    other = filler[:2]
    full_board = board + filler[2:2 + 5 - len(board)]
    holes = (mine, other) if seat == 0 else (other, mine)
    state = HandState(holes[0], holes[1], full_board)
    action = resp.get("action") or ""
    for street_actions in action.split("/"):
        i = 0
        while i < len(street_actions):
            ch = street_actions[i]
            if ch == "b":
                j = i + 1
                while j < len(street_actions) and street_actions[j].isdigit():
                    j += 1
                token = street_actions[i:j]
                i = j
            else:
                token = ch
                i += 1
            if state.over:
                raise ValueError(f"action continues after the hand ended: {action!r}")
            state.apply(token)        # raises ValueError if the move breaks our rules
    return state


def safe_action(bot, state, seat):
    """The bot's move, double checked against the rules before it is sent."""
    incr = bot.act(state, seat)
    if state.legal(incr):
        return incr, False
    fallback = "k" if state.legal("k") else "c"
    return fallback, True


def load_previous(path):
    """Winnings already in the log. A half written last line (from a hard crash) is ignored."""
    results = []
    if not path.exists():
        return results
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                results.append(float(row["winnings"]))
            except (KeyError, TypeError, ValueError):
                pass
    return results


def summary(results, aborted, illegal, client, bot):
    if not results:
        return "no hands finished yet"
    arr = np.array(results, float)
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    return (f"{len(arr):>6} hands  {arr.mean():+8.1f} bb/100  (95% CI +/- {1.96 * se:.1f})  "
            f"abandoned {aborted}  fixed illegal moves {illegal}  network retries {client.network_retries}  "
            f"bot fallbacks {bot.fallbacks}/{bot.decisions}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=500, help="total hands to reach (including resumed ones)")
    ap.add_argument("--bot", default=str(HERE / "models" / "holdem_bot.npz"))
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--resume", action="store_true", help="keep the hands already in the log and add to them")
    ap.add_argument("--host", default=DEFAULT_HOST, help="server address, only change this for testing")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-wait", type=float, default=60.0, help="longest pause between retries, in seconds")
    ap.add_argument("--log-dir", default=str(HERE / "results"))
    ap.add_argument("--delay", type=float, default=0.05, help="pause between hands, in seconds")
    ap.add_argument("--search", action="store_true", help="solve river spots in real time (about 1s per river decision)")
    ap.add_argument("--turn-search", action="store_true", help="solve turn and river spots in real time (about 2s per turn decision)")
    args = ap.parse_args(argv)

    from poker_cfr.holdem.agents import CFRAgent
    from poker_cfr.holdem.search import SearchAgent
    from poker_cfr.holdem.turn_search import TurnSearchAgent

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path, err_path = log_dir / "slumbot_log.csv", log_dir / "slumbot_errors.log"

    def say(msg):
        print(msg, flush=True)

    def record_error(kind, err, resp=None):
        with err_path.open("a", encoding="utf-8") as ef:
            ef.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {kind}: {err}\n")
            if resp is not None:
                ef.write(f"last reply: {json.dumps(resp)[:2000]}\n")
            ef.write(traceback.format_exc() + "\n")

    if not Path(args.bot).exists():
        say(f"can't find the trained bot at {args.bot}")
        return 1
    bot = (TurnSearchAgent if args.turn_search else SearchAgent if args.search else CFRAgent).from_file(args.bot)
    client = Client(args.host, args.timeout, max_wait=args.max_wait, log=say)

    def login():
        if not args.username:
            return None
        return client.post("/api/login", {"username": args.username, "password": args.password})["token"]

    try:
        token = login()
    except Exception as e:
        say(f"login failed: {e}")
        return 1

    results = load_previous(log_path) if args.resume else []
    if args.resume:
        say(f"resuming with {len(results)} hands already logged")
    write_header = not (args.resume and log_path.exists() and results)
    aborted = illegal = in_a_row = 0

    with log_path.open("w" if write_header else "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["hand", "winnings", "baseline_winnings", "action", "our_cards", "board"])
        try:
            while len(results) < args.hands:
                resp = None
                try:
                    resp = client.post("/api/new_hand", {"token": token} if token else {})
                    token = resp.get("token", token)
                    seat = 1 - int(resp["client_pos"])       # client_pos 0 = big blind = our seat 1
                    bot.new_hand()
                    while "winnings" not in resp:
                        state = rebuild(resp, seat)
                        if state.over or state.to_act != seat:
                            raise ValueError(f"server is waiting on us but by our count it isn't our turn: "
                                             f"{resp.get('action')!r}")
                        incr, fixed = safe_action(bot, state, seat)
                        illegal += fixed
                        resp = client.post("/api/act", {"token": token, "incr": incr}, idempotent=False)
                        token = resp.get("token", token)
                    winnings = float(resp["winnings"])
                except SessionLost as e:
                    say(f"  session expired ({e}), starting a fresh one")
                    record_error("session", e, resp)
                    token = login()
                    aborted += 1
                    in_a_row += 1
                except (HandAborted, ValueError, KeyError, TypeError) as e:
                    aborted += 1
                    in_a_row += 1
                    record_error("hand abandoned", e, resp)
                    say(f"  hand abandoned ({e}). Not counted, details in {err_path.name}")
                else:
                    in_a_row = 0
                    results.append(winnings)
                    w.writerow([len(results), winnings, resp.get("baseline_winnings", ""), resp.get("action", ""),
                                " ".join(resp.get("hole_cards") or []), " ".join(resp.get("board") or [])])
                    f.flush()
                    if len(results) % 50 == 0 or len(results) == args.hands:
                        say(summary(results, aborted, illegal, client, bot))
                if in_a_row >= MAX_ABORTS_IN_A_ROW:
                    say(f"stopping: {in_a_row} hands in a row failed. Check your connection or "
                        f"{err_path}, then run again with --resume")
                    break
                time.sleep(args.delay)
        except KeyboardInterrupt:
            say("\nstopped by you. Finished hands are saved, carry on later with --resume")
    say("final: " + summary(results, aborted, illegal, client, bot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
