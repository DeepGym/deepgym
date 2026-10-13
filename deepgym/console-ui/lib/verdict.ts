/** Pure helpers behind the console's verdict views: statuses, colour bands, headlines and the activity feed. */

import { formatScore } from './format';
import type { CaseResult, QAReport, QAStatus, RunStatus, RunSummary } from './types';

export type Tone = 'good' | 'warn' | 'bad';

/** Colour band of a value in [0, 1] (a score or a pass rate): good from 0.8, warn from 0.5, bad below. */
export const scoreTone = (value: number): Tone => (value >= 0.8 ? 'good' : value >= 0.5 ? 'warn' : 'bad');

/** Whether a run is still queued or running. */
export const isActive = <T extends { status: RunStatus }>(
  run: T,
): run is T & { status: 'queued' | 'running' } => run.status === 'queued' || run.status === 'running';

export type TaskStatus = 'failed' | 'inconclusive' | 'running' | 'queued' | 'done' | 'skipped';

/** A run's row status: in flight as is, otherwise its pass/fail verdict. */
export const runTaskStatus = (run: RunSummary): TaskStatus =>
  isActive(run) ? run.status : run.passed ? 'done' : 'failed';

/** The row status of a QA check's or report's verdict. */
export const QA_TASK_STATUS: Record<QAStatus, TaskStatus> = {
  pass: 'done',
  fail: 'failed',
  inconclusive: 'inconclusive',
  skip: 'skipped',
};

const ATTENTION: Record<TaskStatus, number> = {
  failed: 0,
  inconclusive: 1,
  running: 2,
  queued: 3,
  done: 4,
  skipped: 5,
};

/** Rows that need attention first: failed, inconclusive, running, queued, passed, then skipped; order kept within each. */
export const failuresFirst = <T extends { status: TaskStatus }>(rows: T[]): T[] =>
  rows.toSorted((a, b) => ATTENTION[a.status] - ATTENTION[b.status]);

/**
 * Headline for a verdict banner, e.g. "QA failed: 2 of 6 checks" or "QA passed: 6 of 6 checks".
 * `noun` is singular; it gets an "s" when `total` isn't 1.
 */
export function verdictLine(
  subject: string,
  passed: boolean,
  failed: number,
  total: number,
  noun: string,
): string {
  const verdict = `${subject} ${passed ? 'passed' : 'failed'}`;
  if (total === 0) return verdict;
  const things = `${total} ${noun}${total === 1 ? '' : 's'}`;
  if (passed) return `${verdict}: ${total - failed} of ${things}`;
  return failed === 0 ? `${verdict}: every ${noun} passed` : `${verdict}: ${failed} of ${things}`;
}

/** Why a case passed or failed, in one line: its error, else what was expected and what came back. */
export function caseReason(item: CaseResult): string {
  if (item.error) return item.error;
  if (item.passed || !item.expected_summary) return item.actual_summary;
  return `expected ${item.expected_summary}, got ${item.actual_summary || 'nothing'}`;
}

export interface ActivityItem {
  key: string;
  /** When the run started or the QA report was written (ISO). */
  at: string;
  label: string;
  subline: string;
  status: TaskStatus;
  href: string;
}

function runSubline(run: RunSummary): string {
  if (run.status === 'queued') return 'Run queued';
  if (run.status === 'running') return 'Run in progress';
  return run.error_type ? `Run error: ${run.error_type}` : `Run scored ${formatScore(run.score)}`;
}

/** The latest runs and QA verdicts, newest first; QA reports of environments it can't check are left out. */
export function recentActivity(runs: RunSummary[], reports: QAReport[], limit: number): ActivityItem[] {
  const items: ActivityItem[] = [
    ...runs.map((run) => ({
      key: `run-${run.id}`,
      at: run.created_at,
      label: run.environment,
      subline: runSubline(run),
      status: runTaskStatus(run),
      href: `/runs/detail/?id=${run.id}`,
    })),
    ...reports
      .filter((report) => report.status !== 'skip')
      .map(
        (report): ActivityItem => ({
          key: `qa-${report.environment}`,
          at: report.created_at,
          label: report.environment,
          subline: `QA: ${report.checks.filter((check) => check.passed).length} of ${report.checks.length} checks passed`,
          status: QA_TASK_STATUS[report.status],
          href: `/qa/?env=${encodeURIComponent(report.environment)}`,
        }),
      ),
  ];
  // Runs and QA reports format their timestamps differently, so compare instants, not strings.
  return items.toSorted((a, b) => Date.parse(b.at) - Date.parse(a.at)).slice(0, limit);
}
