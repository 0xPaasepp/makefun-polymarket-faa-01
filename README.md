# FAA Decay Forecast

Entry for the make.fun `polymarket-faa-attribution-01` competition: forecasting
the four FAA outage attribution markets (which cause, if any, the FAA outage
will be attributed to) over a 240-minute horizon, five quantiles per minute,
scored by pinball loss.

## Model

Each market's Yes price is forecast as decaying exponentially toward zero at
a rate taken from its own recent fall, floored at the median decay rate of
the four markets as a family — every market crashed 85–96% in its first few
hours, so decay, not reversion, is the centre of the forecast. There is no
mean reversion toward the price's share of the four-market total, because the
four prices sum to about 0.10 rather than 1.0 (the dominant "none of these"
outcome has no market). Around the decaying centre, uncertainty comes from a
detrended EWMA volatility (short and long halflives), spread with a
Student-t(3) jump distribution and horizon scaling of h^0.41, and mixed with
a point mass at the centre for the chance the price simply does not move.
The spread is asymmetric — 1.3x upward, 0.8x downward — because prices near
zero have little room to fall but can spike sharply on news. Volatility is
capped at 0.008 so the crash itself cannot blow the quantiles out.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit input dir> MAKEFUN_OUTPUT_DIR=out node bot/main.ts
```

Node 22, standard library only, no network. Reads every `*.in` case in the
input directory and writes one `<case-id>.out` file per case.

## Practice score

Pinball loss **0.286** on the kit practice cases, versus a **0.364** baseline.
