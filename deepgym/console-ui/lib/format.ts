/** Display helpers shared by tables and detail views. */

const relative = new Intl.RelativeTimeFormat('en', { numeric: 'auto', style: 'short' });
const STEPS: [Intl.RelativeTimeFormatUnit, number][] = [
  ['day', 86_400],
  ['hour', 3_600],
  ['minute', 60],
];

/** "3 min. ago" style relative time for an ISO timestamp. */
export function timeAgo(iso: string, now = Date.now()): string {
  const seconds = (new Date(iso).getTime() - now) / 1000;
  for (const [unit, size] of STEPS) {
    if (Math.abs(seconds) >= size) return relative.format(Math.round(seconds / size), unit);
  }
  return 'just now';
}

/** Compact duration: 840ms, 3.2s, 4m 05s. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '-';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes}m ${String(Math.round((ms % 60_000) / 1000)).padStart(2, '0')}s`;
}

/** Score with two decimals, or a dash when unknown. */
export function formatScore(score: number | null | undefined): string {
  return score === null || score === undefined ? '-' : score.toFixed(2);
}

/** A fraction in [0, 1] as a whole percent, or a dash when unknown. */
export function formatPercent(fraction: number | null): string {
  return fraction === null ? '-' : `${Math.round(fraction * 100)}%`;
}

/** Case-insensitive match of `query` against any of `fields`. */
export function matches(query: string, ...fields: (string | undefined)[]): boolean {
  const needle = query.trim().toLowerCase();
  return !needle || fields.some((field) => field?.toLowerCase().includes(needle));
}
