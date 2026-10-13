import { expect, test } from 'bun:test';
import type { CaseResult } from './types';
import { caseReason, failuresFirst, scoreTone, verdictLine } from './verdict';

test('failures first, then in-flight, then passed, keeping order within each status', () => {
  const rows = [
    { key: 'a', status: 'done' },
    { key: 'b', status: 'failed' },
    { key: 'c', status: 'running' },
    { key: 'd', status: 'failed' },
    { key: 'e', status: 'queued' },
  ] as const;
  expect(failuresFirst([...rows]).map((row) => row.key)).toEqual(['b', 'd', 'c', 'e', 'a']);
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
