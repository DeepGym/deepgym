'use client';
// Adapted from Beautiful UI "Task Rows" (https://beautifului.dev), MIT. See ./LICENSE.
// Changes: driven by real statuses instead of a demo timeline, links per row, inline detail sublines, BoardUI tokens.

import { RiCheckLine, RiCloseLine } from '@remixicon/react';
import Link from 'next/link';
import type { ReactNode } from 'react';
import type { TaskStatus } from '@/lib/verdict';
import { cx } from '@/utils/cx';

export interface TaskRow {
  key: string;
  label: ReactNode;
  /** One line under the label, e.g. a metric or a check's details; shown in full when the row failed. */
  subline?: string;
  /** Right-aligned mono value, e.g. a score. */
  meta?: string;
  status: TaskStatus;
  href?: string;
}

function Ring({ active }: { active: boolean }) {
  return (
    <span className="relative inline-flex size-5 shrink-0">
      <svg aria-hidden viewBox="0 0 20 20" className={cx('size-5', active && 'animate-spin')}>
        <circle cx="10" cy="10" r="8" fill="none" stroke="var(--color-separator-border)" strokeWidth="2" />
        {active && (
          <circle
            cx="10"
            cy="10"
            r="8"
            fill="none"
            stroke="var(--color-accent-400)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray="14 36"
          />
        )}
      </svg>
    </span>
  );
}

function StatusMark({ status }: { status: TaskStatus }) {
  if (status === 'running' || status === 'queued') return <Ring active={status === 'running'} />;
  const done = status === 'done';
  const Icon = done ? RiCheckLine : RiCloseLine;
  return (
    <span
      className={cx(
        'flex size-5 shrink-0 items-center justify-center rounded-full text-text-white',
        done ? 'bg-status-lime-text' : 'bg-status-rose-text',
      )}
    >
      <Icon aria-hidden className="size-3.5" />
    </span>
  );
}

const PILL: Record<TaskStatus, { text: string; className: string }> = {
  done: { text: 'Passed', className: 'bg-status-lime-background text-status-lime-text' },
  failed: { text: 'Failed', className: 'bg-status-rose-background text-status-rose-text' },
  running: { text: 'Running', className: 'bg-status-blue-background text-status-blue-text' },
  queued: { text: 'Queued', className: 'bg-background-tertiary-default text-text-secondary' },
};

function RowBody({ row }: { row: TaskRow }) {
  return (
    <>
      <StatusMark status={row.status} />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5 pt-px">
        <span className="truncate text-body-2-medium text-text-primary">{row.label}</span>
        {row.subline && (
          <span
            className={cx(
              'text-caption-1-medium',
              row.status === 'failed'
                ? 'break-words whitespace-pre-wrap text-text-error-primary'
                : 'truncate text-text-tertiary',
            )}
          >
            {row.subline}
          </span>
        )}
      </span>
      {row.meta && (
        <span className="font-mono text-caption-1-medium leading-5 text-text-secondary tabular-nums">
          {row.meta}
        </span>
      )}
      <span
        className={cx(
          'inline-flex h-5 shrink-0 items-center rounded-full px-2 text-caption-1-medium',
          PILL[row.status].className,
        )}
      >
        {PILL[row.status].text}
      </span>
    </>
  );
}

/** Compact status list: one row per task, with its verdict on the right and any detail inline. */
export function TaskRows({ rows, className }: { rows: TaskRow[]; className?: string }) {
  const rowClass =
    'flex min-h-10 w-full items-start gap-2.5 px-3 py-2.5 text-start outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-border-focus-ring';

  return (
    <div
      className={cx(
        'flex w-full flex-col overflow-hidden rounded-2xl border border-border-button-default bg-background-primary-default shadow-card',
        className,
      )}
    >
      {rows.map((row, index) => (
        <div
          key={row.key}
          className="border-b border-separator-border transition-colors last:border-0 hover:bg-background-secondary-hover"
          style={{
            animation: `fade-up 400ms cubic-bezier(0.23,1,0.32,1) ${Math.min(index, 10) * 40}ms both`,
          }}
        >
          {row.href ? (
            <Link href={row.href} className={rowClass}>
              <RowBody row={row} />
            </Link>
          ) : (
            <div className={rowClass}>
              <RowBody row={row} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
