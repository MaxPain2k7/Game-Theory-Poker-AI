## Fold frequency at a few spots (nit, 3,000 hands)

| spot | true | full histories | showdown hands only | EM over hidden cards |
|---|---|---|---|---|
| J facing a bet, first round | 0.85 | 0.86 | 0.05 | 0.81 |
| Q facing a bet, first round | 0.20 | 0.23 | 0.00 | 0.24 |
| J facing a bet after checking | 0.85 | 0.87 | 0.07 | 0.85 |
| Q facing a bet after checking | 0.20 | 0.20 | 0.00 | 0.17 |

Only 47% of hands reached showdown.

## Best response to the learned model, EV against the real opponent (mean of 3 seeds)

### nit (Nash gets +0.083, ceiling +0.549)

| hands | full histories | showdown hands only | EM over hidden cards |
|---|---|---|---|
| 300 | +0.360 | +0.176 | +0.403 |
| 1,000 | +0.351 | +0.181 | +0.392 |
| 3,000 | +0.442 | +0.233 | +0.459 |
| 10,000 | +0.488 | +0.308 | +0.489 |

### lag (Nash gets +0.369, ceiling +1.249)

| hands | full histories | showdown hands only | EM over hidden cards |
|---|---|---|---|
| 300 | +0.031 | +0.198 | +0.196 |
| 1,000 | +0.453 | +0.505 | +0.277 |
| 3,000 | +0.800 | +0.630 | +0.732 |
| 10,000 | +0.886 | +0.787 | +0.816 |

## Robust responses on a model learned from 1,000 showdown-only hands

risk = worst case loss per hand, edge = EV per hand against the real opponent

### vs nit

| method | p=0.0 | p=0.1 | p=0.25 | p=0.4 | p=0.55 | p=0.7 | p=0.85 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| RNR, true strategy (ceiling) | +0.083 @ 0.000 | +0.124 @ 0.002 | +0.149 @ 0.008 | +0.165 @ 0.016 | +0.208 @ 0.061 | +0.305 @ 0.243 | +0.459 @ 0.755 | +0.549 @ 3.600 |
| RNR, EM model | +0.083 @ 0.000 | +0.105 @ 0.002 | +0.128 @ 0.009 | +0.138 @ 0.017 | +0.150 @ 0.039 | +0.223 @ 0.192 | +0.323 @ 0.588 | +0.369 @ 2.526 |
| DBR, EM model | +0.083 @ 0.000 | +0.084 @ 0.004 | +0.100 @ 0.013 | +0.111 @ 0.020 | +0.119 @ 0.031 | +0.127 @ 0.045 | +0.130 @ 0.055 | +0.151 @ 0.095 |
| RNR, showdown-only model | +0.083 @ 0.000 | +0.102 @ 0.005 | +0.095 @ 0.022 | +0.108 @ 0.058 | +0.129 @ 0.164 | +0.145 @ 0.335 | +0.146 @ 0.375 | +0.159 @ 2.117 |

### vs lag

| method | p=0.0 | p=0.1 | p=0.25 | p=0.4 | p=0.55 | p=0.7 | p=0.85 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| RNR, true strategy (ceiling) | +0.369 @ 0.000 | +0.541 @ 0.008 | +0.841 @ 0.074 | +1.003 @ 0.147 | +1.120 @ 0.245 | +1.158 @ 0.318 | +1.228 @ 0.533 | +1.249 @ 0.800 |
| RNR, EM model | +0.369 @ 0.000 | +0.495 @ 0.008 | +0.611 @ 0.050 | +0.761 @ 0.113 | +0.837 @ 0.207 | +0.807 @ 0.392 | +0.464 @ 1.018 | +0.182 @ 2.500 |
| DBR, EM model | +0.369 @ 0.000 | +0.397 @ 0.006 | +0.451 @ 0.015 | +0.495 @ 0.038 | +0.545 @ 0.045 | +0.593 @ 0.061 | +0.647 @ 0.095 | +0.701 @ 0.129 |
| RNR, showdown-only model | +0.369 @ 0.000 | +0.501 @ 0.008 | +0.704 @ 0.085 | +0.778 @ 0.145 | +0.748 @ 0.289 | +0.734 @ 0.452 | +0.548 @ 1.013 | +0.305 @ 2.114 |

