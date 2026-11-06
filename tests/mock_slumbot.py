"""A local stand-in for slumbot.com that misbehaves on purpose.

It speaks the same protocol (POST /api/new_hand, POST /api/act), plays the
other seat with a random bettor and keeps its own record of every hand's
result so a test can check the client's log against the truth. Faults are
injected at random:

  drop        close the connection before reading the request
  lost_reply  apply the request, then close the connection without answering
  error500    answer HTTP 500 with an HTML page
  garbage     answer 200 with something that isn't JSON
  slow        wait longer than the client's timeout before answering
  expire      forget the session and answer with an invalid token error
"""

from __future__ import annotations

import json
import random
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from poker_cfr.holdem.agents import RandomBot
from poker_cfr.holdem.cards import name
from poker_cfr.holdem.engine import HandState


class MockSlumbot:
    def __init__(self, faults=None, slow_seconds=1.5, seed=0):
        self.faults = faults or {}
        self.slow_seconds = slow_seconds
        self.rng = random.Random(seed)
        self.opponent = RandomBot(seed)
        self.sessions = {}
        self.delivered = []        # winnings of hands whose final reply reached the client
        self.lock = threading.Lock()
        self.fault_counts = {k: 0 for k in ("drop", "lost_reply", "error500", "garbage", "slow", "expire")}
        handler = self._make_handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    # ---------------------------------------------------------------- game logic
    def _advance(self, sess):
        st = sess["state"]
        while not st.over and st.to_act != sess["seat"]:
            st.apply(self.opponent.act(st, st.to_act))

    def _reply(self, token, sess):
        st = sess["state"]
        out = {"token": token, "action": st.history, "client_pos": 1 - sess["seat"],
               "hole_cards": [name(c) for c in st.hole[sess["seat"]]],
               "board": [name(c) for c in st.board]}
        if st.over:
            win = st.payoff() * (1 if sess["seat"] == 0 else -1)
            out.update(winnings=win, baseline_winnings=0,
                       bot_hole_cards=[name(c) for c in st.hole[1 - sess["seat"]]])
        return out

    def new_hand(self, payload):
        token = payload.get("token") or secrets.token_hex(8)
        if payload.get("token") and token not in self.sessions:
            return {"error_msg": "Invalid token"}
        cards = self.rng.sample(range(52), 9)
        sess = {"state": HandState(cards[0:2], cards[2:4], cards[4:9]), "seat": self.rng.randrange(2)}
        self.sessions[token] = sess
        self._advance(sess)
        return self._reply(token, sess)

    def act(self, payload):
        token = payload.get("token")
        sess = self.sessions.get(token)
        if sess is None:
            return {"error_msg": "Invalid token"}
        st = sess["state"]
        incr = payload.get("incr", "")
        if st.over or st.to_act != sess["seat"] or not st.legal(incr):
            return {"error_msg": f"Illegal action {incr!r}"}
        st.apply(incr)
        self._advance(sess)
        return self._reply(token, sess)

    # ---------------------------------------------------------------- http side
    def _make_handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _roll(self, kind):
                if mock.rng.random() < mock.faults.get(kind, 0.0):
                    mock.fault_counts[kind] += 1
                    return True
                return False

            def do_POST(self):
                with mock.lock:
                    if self._roll("drop"):
                        self.close_connection = True
                        self.connection.close()
                        return
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if self._roll("error500"):
                        return self._send(500, "<html>Internal Server Error</html>", "text/html")
                    if self._roll("garbage"):
                        return self._send(200, "<html>maintenance</html>", "text/html")
                    if payload.get("token") in mock.sessions and self._roll("expire"):
                        del mock.sessions[payload["token"]]
                    slow = self._roll("slow")
                    if self.path == "/api/new_hand":
                        out = mock.new_hand(payload)
                    elif self.path == "/api/act":
                        out = mock.act(payload)
                    else:
                        return self._send(404, json.dumps({"error_msg": "Not found"}))
                    lost = self._roll("lost_reply")
                    if "winnings" in out and not lost and not slow:
                        mock.delivered.append(out["winnings"])
                if slow:
                    time.sleep(mock.slow_seconds)
                    return
                if lost:
                    self.close_connection = True
                    self.connection.close()
                    return
                self._send(200, json.dumps(out))

        return Handler
