# Local Best Response against the hold'em bot

LBR (Lisy and Bowling 2017) sees the bot's blueprint strategy, tracks its range and picks the best local action at each of its own decisions. It checks or calls preflop and on the flop and bets pot or all-in on the turn and river, the paper's strongest setup. What it wins is a lower bound on the bot's exploitability.

Bot: models/holdem_bot.npz (blueprint, no search). Duplicate dealing, blocks of 500 deals (2,000 hands).

| seed | block | hands | LBR wins (bb/100) | std error |
|---|---|---|---|---|
| 1 | 0 | 2,000 | +271.1 | 41.2 |
| 1 | 1 | 2,000 | -21.4 | 52.9 |
| 1 | 2 | 2,000 | +115.7 | 46.8 |
| 1 | 3 | 2,000 | +134.9 | 57.0 |
| 1 | 4 | 2,000 | +66.9 | 52.8 |
| 1 | 5 | 2,000 | +174.3 | 50.7 |
| 1 | 6 | 2,000 | +141.4 | 56.1 |
| 1 | 7 | 2,000 | +92.1 | 55.1 |
| 2 | 0 | 2,000 | +84.1 | 57.0 |
| 2 | 1 | 2,000 | +166.4 | 46.8 |
| 2 | 2 | 2,000 | +66.2 | 55.9 |
| 2 | 3 | 2,000 | +72.5 | 55.2 |
| 2 | 4 | 2,000 | +191.1 | 53.1 |
| 2 | 5 | 2,000 | +217.0 | 46.2 |
| 2 | 6 | 2,000 | +4.8 | 50.1 |
| 2 | 7 | 2,000 | +156.9 | 48.0 |

**Pooled: LBR wins +120.9 bb/100 over 32,000 hands (95% CI +95.5 to +146.2), about 1,209 mbb/hand.**

For scale, the paper's LBR in the same setup beat Slumbot 2016 by 4,020 mbb/hand. Folding every hand loses 750 mbb/hand.

Sanity check: against a bot that always calls, LBR wins about +4,900 bb/100, close to the paper's ~50 big blinds per hand for always-call.
