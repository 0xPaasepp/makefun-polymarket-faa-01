# FAA Decay Forecast — polymarket-faa-attribution-01

Forecasts the 4 FAA outage attribution markets (5 quantiles × 240 min per
checkpoint) with pinball loss as the score.

## Model

Per-market EWMA volatility detrended by the family log-decay, a point-mass
mixture over a Student-t(3) jump distribution, h^0.41 horizon scaling, a
level vol floor, and an adaptive decay of the center toward 0. No mean
reversion to x/sum: the four Yes prices sum to ~0.10, not ~1.0, because the
dominant outcome ("no attribution to any of these") has no market.

- Asymmetry up 1.3x / down 0.8x: prices sit near 0, so a news break can
  spike a market +0.2 in a minute while down moves are bounded by the price.
- SIGMA_MAX 0.008: a wider cap lets a crash inflate sigma and spreads the
  quantiles far too wide around a center that is already moving.
- Practice pinball: **0.286** vs house baseline **0.364** on the kit
  practice cases.

## Run

```sh
MAKEFUN_INPUT_DIR=polymarket-faa-attribution-01-kit/input MAKEFUN_OUTPUT_DIR=out node bot/main.ts
```

Node 22, stdlib only, no network. Runs well under the 8,000 ms per-case cap.
