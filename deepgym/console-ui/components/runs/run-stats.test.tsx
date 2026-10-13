import { afterEach, expect, test } from 'bun:test';
import type { RunStats, RunTotals } from '@/lib/types';
import { cleanup, render } from '@/test/dom';
import { KpiTiles } from './run-stats';

afterEach(cleanup);

/** Ten finished runs, `passed` of them passing and `noVerdict` without a verdict. */
const totals = (passed: number, noVerdict: number): RunTotals => ({
  runs: 10,
  finished: 10,
  passed,
  errors: noVerdict,
  mean_score: passed / 10,
  mean_duration_ms: 100,
  judged_count: 10 - noVerdict,
  no_verdict_count: noVerdict,
  judged_mean_score: passed / (10 - noVerdict),
});

const stats = (current: RunTotals, previous: RunTotals | null): RunStats => ({
  hours: 168,
  current,
  previous,
  score_bins: Array<number>(10).fill(0),
  error_types: [],
  environments: [],
});

test('the judged figures and the no-verdict share stay visible next to deltas', async () => {
  const { container } = await render(<KpiTiles stats={stats(totals(6, 2), totals(5, 0))} range="7d" />);

  const text = container.textContent ?? '';
  expect(text).toContain('vs prev 7d');
  expect(text).toContain('60%'); // the headline counts the two as failed
  expect(text).toContain('judged 75%');
  expect(text).toContain('judged 0.75');
  expect(text).toContain('20% of finished');
});

test('no judged line when every finished run got a verdict', async () => {
  const { container } = await render(<KpiTiles stats={stats(totals(6, 0), null)} range={null} />);

  expect(container.textContent).not.toContain('judged');
  expect(container.textContent).toContain('6 passed');
});
