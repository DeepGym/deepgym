/** Pure helpers for the stats views: time ranges, pass rates, deltas, environment health and ETAs. */

import { formatDuration } from './format';
import type { RunSummary, RunTotals } from './types';

/** Time ranges the stats can cover, in hours. */
export const RANGES = { '24h': 24, '7d': 168, '30d': 720 } as const;
export type Range = keyof typeof RANGES;
export const DEFAULT_RANGE: Range = '7d';

/** The range named in the URL, or the default when it's missing or unknown. */
export const parseRange = (value: string | null): Range =>
  value !== null && Object.hasOwn(RANGES, value) ? (value as Range) : DEFAULT_RANGE;

/** Share of finished runs that passed, or null before any finished. */
export const passRate = (totals: RunTotals): number | null =>
  totals.finished ? totals.passed / totals.finished : null;

export interface Delta {
  /** Signed change, e.g. "+12%", "-5 pts", "+0.04", or "no change". */
  change: string;
  /** Whether the change is an improvement; neutral for no change or metrics with no better direction. */
  tone: 'good' | 'bad' | 'neutral';
}

/** How a metric changes: counts by percent, rates (0-1) by percentage points, scores by value. */
export type DeltaUnit = 'count' | 'rate' | 'score';
/** Which direction is an improvement. */
export type Better = 'higher' | 'lower' | 'neither';

const signed = (value: string, positive: boolean) => (positive ? `+${value}` : value);

function describeChange(current: number, previous: number, unit: DeltaUnit): string | null {
  const diff = current - previous;
  if (unit === 'rate') {
    const points = Math.round(diff * 100);
    return points === 0 ? null : `${signed(String(points), points > 0)} pts`;
  }
  if (unit === 'score') return Math.abs(diff) < 0.005 ? null : signed(diff.toFixed(2), diff > 0);
  if (diff === 0) return null;
  // From zero there is no percentage, so show the absolute change.
  if (previous === 0) return signed(String(diff), diff > 0);
  return `${signed(String(Math.round((diff / previous) * 100)), diff > 0)}%`;
}

/**
 * Change from `previous` to `current`, coloured by whether it is an improvement rather than by
 * its sign (more errors is bad, a higher pass rate good). Null when either side is unknown.
 */
export function delta(
  current: number | null,
  previous: number | null,
  unit: DeltaUnit,
  better: Better,
): Delta | null {
  if (current === null || previous === null) return null;
  const change = describeChange(current, previous, unit);
  if (change === null) return { change: 'no change', tone: 'neutral' };
  if (better === 'neither') return { change, tone: 'neutral' };
  return { change, tone: current > previous === (better === 'higher') ? 'good' : 'bad' };
}

/** Median duration of the finished runs given, or null when none has one. */
export function medianDuration(runs: RunSummary[]): number | null {
  const durations = runs
    .flatMap((run) => (run.duration_ms === null ? [] : [run.duration_ms]))
    .toSorted((a, b) => a - b);
  if (!durations.length) return null;
  const middle = durations.length >> 1;
  return durations.length % 2 ? durations[middle] : (durations[middle - 1] + durations[middle]) / 2;
}

/** What to expect for a run `elapsedMs` in, given how long earlier runs of it took (null with no history). */
export function etaNote(elapsedMs: number, typicalMs: number | null): string {
  if (typicalMs === null) return 'No earlier runs of this environment to estimate from';
  const left = typicalMs - elapsedMs;
  return left > 0
    ? `About ${formatDuration(left)} left`
    : `Taking longer than usual (typically ${formatDuration(typicalMs)})`;
}

/** Environments needing attention first: lowest pass rate, those with no finished run last; then most runs. */
export function worstFirst<T extends RunTotals>(rows: T[]): T[] {
  const rank = (row: T) => passRate(row) ?? Number.POSITIVE_INFINITY;
  return rows.toSorted((a, b) => rank(a) - rank(b) || b.runs - a.runs);
}
