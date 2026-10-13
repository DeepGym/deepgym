/** Pure helpers behind the console's verdict views: score colour bands, case reasons and headlines. */

import type { CaseResult } from './types';

export type Tone = 'good' | 'warn' | 'bad';

/** Colour band of a value in [0, 1] (a score or a pass rate): good from 0.8, warn from 0.5, bad below. */
export const scoreTone = (value: number): Tone => (value >= 0.8 ? 'good' : value >= 0.5 ? 'warn' : 'bad');

/** Why a case passed or failed, in one line: its error, else what was expected and what came back. */
export function caseReason(item: CaseResult): string {
  if (item.error) return item.error;
  if (item.passed || !item.expected_summary) return item.actual_summary;
  return `expected ${item.expected_summary}, got ${item.actual_summary || 'nothing'}`;
}
