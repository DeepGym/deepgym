import { expect, test } from 'bun:test';
import { delta, parseRange, passRate, worstFirst } from './stats';
import type { RunTotals } from './types';

const totals = (runs: number, finished: number, passed: number): RunTotals => ({
  runs,
  finished,
  passed,
  errors: 0,
  mean_score: null,
  mean_duration_ms: null,
});

test('deltas are coloured by improvement, not by sign', () => {
  expect(delta(0.8, 0.7, 'rate', 'higher')).toEqual({ change: '+10 pts', tone: 'good' });
  expect(delta(0.6, 0.7, 'rate', 'higher')).toEqual({ change: '-10 pts', tone: 'bad' });
  expect(delta(12, 10, 'count', 'lower')).toEqual({ change: '+20%', tone: 'bad' });
  expect(delta(8, 10, 'count', 'lower')).toEqual({ change: '-20%', tone: 'good' });
  expect(delta(112, 100, 'count', 'neither')).toEqual({ change: '+12%', tone: 'neutral' });
  expect(delta(0.9, 0.85, 'score', 'higher')).toEqual({ change: '+0.05', tone: 'good' });
});

test('no change, a zero baseline and unknown sides', () => {
  expect(delta(5, 5, 'count', 'lower')).toEqual({ change: 'no change', tone: 'neutral' });
  expect(delta(0.701, 0.7, 'rate', 'higher')).toEqual({ change: 'no change', tone: 'neutral' });
  expect(delta(3, 0, 'count', 'lower')).toEqual({ change: '+3', tone: 'bad' });
  expect(delta(null, 0.5, 'rate', 'higher')).toBeNull();
  expect(delta(0.5, null, 'rate', 'higher')).toBeNull();
});

test('range from the URL falls back to 7d', () => {
  expect(parseRange('24h')).toBe('24h');
  expect(parseRange('30d')).toBe('30d');
  expect(parseRange(null)).toBe('7d');
  expect(parseRange('toString')).toBe('7d');
});

test('pass rate counts finished runs only', () => {
  expect(passRate(totals(5, 4, 3))).toBe(0.75);
  expect(passRate(totals(2, 0, 0))).toBeNull();
});

test('worst pass rate first, unfinished last, ties by most runs', () => {
  const rows = [
    { ...totals(4, 4, 4), environment: 'healthy' },
    { ...totals(2, 0, 0), environment: 'pending' },
    { ...totals(3, 2, 1), environment: 'half-small' },
    { ...totals(9, 4, 2), environment: 'half-big' },
    { ...totals(5, 5, 0), environment: 'broken' },
  ];
  expect(worstFirst(rows).map((row) => row.environment)).toEqual([
    'broken',
    'half-big',
    'half-small',
    'healthy',
    'pending',
  ]);
});
