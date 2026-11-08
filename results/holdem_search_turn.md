Turn plus river search minus river search only, both against the blueprint bot.
Paired comparison, 200 turn iterations sampling 4 river cards each, 4 independent chunks:

| seed | result | 95% CI | paired hands |
|---|---|---|---|
| 101 | -43.8 bb/100 | -91.6 to +4.1 | 3,000 |
| 102 | +8.1 bb/100 | -46.2 to +62.4 | 3,000 |
| 103 | +7.0 bb/100 | -46.8 to +60.7 | 3,000 |
| 104 | -7.1 bb/100 | -56.2 to +41.9 | 3,000 |
| pooled | **-8.9 bb/100** | **-34.6 to +16.7** | 12,000 |

Solver accuracy on 6 real turn spots (results/holdem_turn_solver_check.txt): the
200 x 4 setting gives up 2.7 chips per decision on average against the best root
action, compared with an exact 400 iteration solve over all 48 river cards.
