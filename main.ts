// FAA outage attribution forecast.
//
// Rationale, in brief: the four Yes prices in each case add up to roughly
// 0.10, not 1.0, because the dominant outcome — the outage being attributed
// to none of these four causes — has no market of its own. So there is no
// reason for a price to revert toward its share of the four-way total; doing
// that would drag every price upward for no reason. Instead each price is
// treated as decaying toward zero: every market collapsed 85-96% in its
// first few hours, and the recent rate of that fall is the best guide to
// where it goes next. The forecast is asymmetric because prices live near
// zero: a price cannot fall below zero, so downside room is bounded by the
// price itself, while a news break can spike a market up by 0.2 in a single
// minute. Volatility is capped tightly (0.008) because a wider cap lets the
// crash itself inflate sigma and spray the quantiles far too wide around a
// centre that is already sliding down.
//
// Model: detrended EWMA volatility (short and long halflives, detrended by
// the decay rate), a point-mass mixture (some chance the price does not move
// at all over the horizon, so the median sits exactly on the decayed centre),
// Student-t(3) jumps for the moving part, decay toward zero at the family
// median rate, and horizon scaling sigma * h^0.41.
//
// Node 22, standard library only, no network, offline.

import * as fs from 'node:fs';
import * as path from 'node:path';

// ---------------------------------------------------------------- constants

const SIGMA_MIN = 0.0003;
const SIGMA_MAX = 0.008;
const HL_SHORT = 8;
const HL_LONG = 1440;
const LONG_K = 0.3;
const LOOKBACK = 360;
const GAMMA = 0.41;
const SQRT_C = 0.005;
const PM_WIN = 120;
const UP_SCALE = 1.3;
const DOWN_SCALE = 0.8;
const DRIFT_WIN = 120;
const DRIFT_MAX = 0.004;
const DRIFT_MIN_MIN = 10;

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];

// Inverse CDF of a Student-t distribution with 3 degrees of freedom, in
// standard-deviation units, tabulated for tau = 0.005, 0.010, ..., 0.995.
const T3: number[] = [
  -3.3723, -2.6216, -2.2494, -2.0103, -1.8374, -1.7035, -1.595, -1.5042,
  -1.4265, -1.3587, -1.2987, -1.2449, -1.1962, -1.1518, -1.111, -1.0733,
  -1.0381, -1.0053, -0.9746, -0.9456, -0.9181, -0.8921, -0.8674, -0.8439,
  -0.8214, -0.7998, -0.7791, -0.7592, -0.74, -0.7216, -0.7037, -0.6864,
  -0.6697, -0.6535, -0.6377, -0.6224, -0.6075, -0.5929, -0.5788, -0.5649,
  -0.5514, -0.5382, -0.5253, -0.5126, -0.5002, -0.4881, -0.4761, -0.4644,
  -0.4529, -0.4416, -0.4305, -0.4195, -0.4088, -0.3981, -0.3877, -0.3774,
  -0.3672, -0.3571, -0.3472, -0.3374, -0.3277, -0.3181, -0.3087, -0.2993,
  -0.29, -0.2808, -0.2717, -0.2627, -0.2538, -0.2449, -0.2361, -0.2274,
  -0.2187, -0.2102, -0.2016, -0.1931, -0.1847, -0.1763, -0.168, -0.1597,
  -0.1515, -0.1433, -0.1351, -0.127, -0.1189, -0.1109, -0.1028, -0.0948,
  -0.0868, -0.0789, -0.0709, -0.063, -0.0551, -0.0472, -0.0393, -0.0314,
  -0.0236, -0.0157, -0.0079, -0.0, 0.0079, 0.0157, 0.0236, 0.0314, 0.0393,
  0.0472, 0.0551, 0.063, 0.0709, 0.0789, 0.0868, 0.0948, 0.1028, 0.1109,
  0.1189, 0.127, 0.1351, 0.1433, 0.1515, 0.1597, 0.168, 0.1763, 0.1847,
  0.1931, 0.2016, 0.2102, 0.2187, 0.2274, 0.2361, 0.2449, 0.2538, 0.2627,
  0.2717, 0.2808, 0.29, 0.2993, 0.3087, 0.3181, 0.3277, 0.3374, 0.3472,
  0.3571, 0.3672, 0.3774, 0.3877, 0.3981, 0.4088, 0.4195, 0.4305, 0.4416,
  0.4529, 0.4644, 0.4761, 0.4881, 0.5002, 0.5126, 0.5253, 0.5382, 0.5514,
  0.5649, 0.5788, 0.5929, 0.6075, 0.6224, 0.6377, 0.6535, 0.6697, 0.6864,
  0.7037, 0.7216, 0.74, 0.7592, 0.7791, 0.7998, 0.8214, 0.8439, 0.8674,
  0.8921, 0.9181, 0.9456, 0.9746, 1.0053, 1.0381, 1.0733, 1.111, 1.1518,
  1.1962, 1.2449, 1.2987, 1.3587, 1.4265, 1.5042, 1.595, 1.7035, 1.8374,
  2.0103, 2.2494, 2.6216, 3.3723,
];

// ------------------------------------------------------------------- helpers

function round6(v: number): number {
  const clamped = Math.min(1, Math.max(0, v));
  return Math.round(clamped * 1e6) / 1e6 + 0;
}

function t3Inv(tau: number): number {
  if (tau <= 0.005) return T3[0];
  if (tau >= 0.995) return T3[T3.length - 1];
  const pos = tau / 0.005 - 1;
  const i = Math.floor(pos);
  const f = pos - i;
  return T3[i] + f * (T3[i + 1] - T3[i]);
}

// EWMA volatility of per-minute price changes, detrended by the decay rate:
// the steady fall is added back before squaring so the crash itself does
// not count as noise. Weights decay exponentially going back in time.
function ewmaVol(
  prices: number[],
  halflife: number,
  lookback: number,
  drift: number,
): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - lookback - 1);
  const alpha = 1 - Math.pow(0.5, 1 / halflife);
  let variance = 0;
  let wsum = 0;
  let w = 1;
  for (let i = n - 2; i >= start; i--) {
    const r = prices[i + 1] - prices[i] + drift * prices[i];
    variance += w * r * r;
    wsum += w;
    w *= 1 - alpha;
  }
  return wsum > 0 ? Math.sqrt(variance / wsum) : 0;
}

function marketSigma(prices: number[], rawDrift: number): number {
  const short = ewmaVol(prices, HL_SHORT, LOOKBACK, rawDrift);
  const long = ewmaVol(prices, HL_LONG, LOOKBACK, rawDrift);
  const x = prices[prices.length - 1];
  return Math.min(
    SIGMA_MAX,
    Math.max(SIGMA_MIN, short, LONG_K * long, SQRT_C * Math.sqrt(x)),
  );
}

// Fraction of recent minutes in which the price moved at all, clipped.
function moveProb(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return 0.1;
  const start = Math.max(0, n - PM_WIN - 1);
  let moves = 0;
  let total = 0;
  for (let i = start; i <= n - 2; i++) {
    total++;
    if (prices[i + 1] !== prices[i]) moves++;
  }
  if (total === 0) return 0.1;
  return Math.min(0.9, Math.max(0.01, moves / total));
}

// This market's own exponential decay rate over the recent window, from the
// log-ratio of the window's first and last prices. Null when there is not
// enough usable history; never negative (growth is reported as no decay).
function ownDecay(prices: number[]): number | null {
  if (prices.length < DRIFT_MIN_MIN + 1) return null;
  const n = prices.length;
  const start = Math.max(0, n - DRIFT_WIN);
  const p0 = prices[start];
  const p1 = prices[n - 1];
  if (p0 <= 0 || p1 <= 0) return null;
  const span = n - 1 - start;
  if (span < DRIFT_MIN_MIN) return null;
  const rate = -Math.log(p1 / p0) / span;
  return rate <= 0 ? 0 : rate;
}

// Median of the markets' own decay rates; 0 when none can be measured.
function familyDecay(markets: { prices: number[] }[]): number {
  const rates: number[] = [];
  for (const m of markets) {
    const r = ownDecay(m.prices);
    if (r !== null) rates.push(r);
  }
  if (rates.length === 0) return 0;
  rates.sort((a, b) => a - b);
  return rates[Math.floor(rates.length / 2)];
}

// ------------------------------------------------------------------ forecast

function forecastMarket(
  id: string,
  prices: number[],
  horizon: number,
  familyDrift: number,
): number[][] {
  if (prices.length === 0) {
    throw new Error(`empty prices for market ${id}`);
  }
  const x = prices[prices.length - 1];
  const own = ownDecay(prices);
  const rawDrift = Math.max(own ?? 0, familyDrift);
  const drift = Math.min(DRIFT_MAX, rawDrift);
  const sigma = marketSigma(prices, rawDrift);
  const pm = moveProb(prices);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const center = x * Math.exp(-drift * h);
    const s = sigma * Math.pow(h, GAMMA);
    const p0 = Math.pow(1 - pm, h);
    const tauLow = (1 - p0) * 0.5;
    const tauHigh = tauLow + p0;
    const row: number[] = [];
    for (const tau of QUANTILES) {
      let q: number;
      if (tau < tauLow) {
        q = center + t3Inv(tau / (1 - p0)) * s * DOWN_SCALE;
      } else if (tau <= tauHigh) {
        q = center;
      } else {
        q = center + t3Inv((tau - p0) / (1 - p0)) * s * UP_SCALE;
      }
      row.push(round6(q));
    }
    // Keep each row non-decreasing left to right.
    for (let j = 1; j < row.length; j++) {
      if (row[j] < row[j - 1]) row[j] = row[j - 1];
    }
    rows.push(row);
  }
  return rows;
}

// ---------------------------------------------------------------------- main

interface MarketInput {
  id: string;
  prices: number[];
}

interface CaseInput {
  horizon_minutes: number;
  quantiles: number[];
  markets: MarketInput[];
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  fs.mkdirSync(outputDir, { recursive: true });

  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith('.in'))
    .sort();

  for (const file of files) {
    const raw = fs.readFileSync(path.join(inputDir, file), 'utf8');
    const kase = JSON.parse(raw) as CaseInput;
    const horizon = kase.horizon_minutes;
    const quantiles = kase.quantiles;
    if (!Number.isInteger(horizon) || horizon <= 0) {
      throw new Error(`unexpected horizon_minutes in ${file}`);
    }
    if (
      !Array.isArray(quantiles) ||
      quantiles.length !== QUANTILES.length ||
      !quantiles.every((q, i) => q === QUANTILES[i])
    ) {
      throw new Error(`unexpected quantiles in ${file}`);
    }
    const markets = kase.markets;
    const fam = familyDecay(markets);
    const forecasts = markets.map((m) => ({
      id: m.id,
      q: forecastMarket(m.id, m.prices, horizon, fam),
    }));
    const caseId = file.slice(0, -3);
    fs.writeFileSync(
      path.join(outputDir, `${caseId}.out`),
      JSON.stringify({ forecasts }) + '\n',
    );
  }
}

main();
