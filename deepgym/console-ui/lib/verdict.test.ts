import { expect, test } from 'bun:test';
import type { CaseResult, QAReport, QAStatus, RunSummary } from './types';
import { caseReason, failuresFirst, recentActivity, scoreTone, verdictLine } from './verdict';

const run = (id: string, created_at: string, extra: Partial<RunSummary> = {}): RunSummary => ({
  id,
  group_id: 'g',
  environment: `env-${id}`,
  mode: 'local',
  provider: 'local',
  reference: true,
  status: 'completed',
  score: 1,
  passed: true,
  error_type: null,
  duration_ms: 10,
  created_at,
  finished_at: created_at,
  ...extra,
});

/** A report with one check whose verdict is also the report's, as `deepgym qa` computes it. */
const report = (environment: string, created_at: string, status: QAStatus): QAReport => ({
  environment,
  status,
  passed: status === 'pass',
  checks: [{ name: 'hack_resistant', status, passed: status === 'pass', details: '' }],
  created_at,
});

test('activity merges runs and QA verdicts newest first, without skipped QA', () => {
  const items = recentActivity(
    [
      run('a', '2026-10-13T10:00:00+00:00'),
      run('b', '2026-10-13T12:00:00.5+00:00', { status: 'failed', passed: false, error_type: 'timeout' }),
      run('c', '2026-10-13T12:30:00+00:00', { status: 'running', score: null, passed: null }),
    ],
    [
      report('qa-env', '2026-10-13T11:00:00Z', 'fail'),
      report('odd-env', '2026-10-13T11:30:00Z', 'inconclusive'),
      report('skip-env', '2026-10-13T13:00:00Z', 'skip'),
    ],
    4,
  );
  expect(items.map((item) => [item.key, item.status, item.subline])).toEqual([
    ['run-c', 'running', 'Run in progress'],
    ['run-b', 'failed', 'Run error: timeout'],
    ['qa-odd-env', 'inconclusive', 'QA: 0 of 1 checks passed'],
    ['qa-qa-env', 'failed', 'QA: 0 of 1 checks passed'],
  ]);
  expect(items[3].href).toBe('/qa/?env=qa-env');
});

test('failures first, then inconclusive, in-flight, passed and skipped, keeping order within each status', () => {
  const rows = [
    { key: 'a', status: 'done' },
    { key: 'b', status: 'failed' },
    { key: 'c', status: 'running' },
    { key: 'd', status: 'failed' },
    { key: 'e', status: 'queued' },
    { key: 'f', status: 'skipped' },
    { key: 'g', status: 'inconclusive' },
  ] as const;
  expect(failuresFirst([...rows]).map((row) => row.key)).toEqual(['b', 'd', 'g', 'c', 'e', 'a', 'f']);
});

test('verdict line counts failures when failed and passes when passed', () => {
  expect(verdictLine('QA', false, 2, 6, 'check')).toBe('QA failed: 2 of 6 checks');
  expect(verdictLine('QA', true, 0, 6, 'check')).toBe('QA passed: 6 of 6 checks');
  expect(verdictLine('Run', true, 2, 10, 'case')).toBe('Run passed: 8 of 10 cases');
  expect(verdictLine('Run', false, 0, 10, 'case')).toBe('Run failed: every case passed');
  expect(verdictLine('Run', false, 1, 1, 'case')).toBe('Run failed: 1 of 1 case');
  expect(verdictLine('Run', false, 0, 0, 'case')).toBe('Run failed');
});

const base: CaseResult = {
  id: 'c1',
  passed: false,
  score: 0,
  input_summary: '[1, 2]',
  expected_summary: '3',
  actual_summary: '4',
  error: null,
  execution_time_ms: 1,
};

test('score bands: good from 0.8, warn from 0.5, bad below', () => {
  expect([1, 0.8, 0.79, 0.5, 0.49, 0].map(scoreTone)).toEqual(['good', 'good', 'warn', 'warn', 'bad', 'bad']);
});

test('case reason prefers the error, then expected vs actual for failures', () => {
  expect(caseReason({ ...base, error: 'ZeroDivisionError' })).toBe('ZeroDivisionError');
  expect(caseReason(base)).toBe('expected 3, got 4');
  expect(caseReason({ ...base, actual_summary: '' })).toBe('expected 3, got nothing');
  expect(caseReason({ ...base, passed: true, actual_summary: '3' })).toBe('3');
  expect(caseReason({ ...base, expected_summary: '' })).toBe('4');
});
