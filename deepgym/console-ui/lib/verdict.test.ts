import { expect, test } from 'bun:test';
import type { CaseResult } from './types';
import { caseReason, scoreTone } from './verdict';

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
