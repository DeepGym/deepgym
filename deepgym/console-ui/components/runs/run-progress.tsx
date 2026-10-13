'use client';

import { Card } from '@/components/ui/card';
import { useApi } from '@/lib/api';
import { formatDuration } from '@/lib/format';
import { etaNote, medianDuration } from '@/lib/stats';
import type { RunDetail, RunSummary } from '@/lib/types';
import { cx } from '@/utils/cx';

export const KEEPS_GOING = 'You can leave this page, the run keeps going.';

/** Median duration of the latest finished runs of `environment` (null while loading or with none). */
export function useTypicalDuration(environment: string | null): number | null {
  const { data } = useApi<RunSummary[]>(
    environment ? `/runs?environment=${encodeURIComponent(environment)}&status=completed&limit=20` : null,
  );
  return data ? medianDuration(data) : null;
}

/**
 * Shown while a run is queued or running: time so far, an estimate from earlier runs of its
 * environment, and that it carries on without this page. Re-renders as the run detail polls.
 * Runs report no per-case progress until they finish, so there is no cases-done count.
 */
export function RunProgress({ run }: { run: RunDetail }) {
  const typical = useTypicalDuration(run.environment);
  const elapsed = Date.now() - Date.parse(run.created_at);
  const fraction = run.status === 'running' && typical ? Math.min(0.95, elapsed / typical) : null;

  return (
    <Card title={run.status === 'queued' ? 'Queued' : 'Running'} className="mt-4">
      <div className="flex flex-col gap-2.5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="text-body-2-medium text-text-primary">
            {run.status === 'queued' ? 'Waiting for a free worker' : `Running for ${formatDuration(elapsed)}`}
          </span>
          <span className="text-body-2-regular text-text-secondary">{etaNote(elapsed, typical)}</span>
        </div>
        <div
          role="progressbar"
          aria-label="Run progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
          aria-valuetext={etaNote(elapsed, typical)}
          className="h-1.5 overflow-hidden rounded-full bg-background-tertiary-default"
        >
          <div
            className={cx(
              'h-full rounded-full bg-accent-400 transition-[width] duration-500',
              fraction === null && 'w-1/4 animate-pulse',
            )}
            style={fraction === null ? undefined : { width: `${fraction * 100}%` }}
          />
        </div>
        <p className="text-caption-1-medium text-text-tertiary">{KEEPS_GOING}</p>
      </div>
    </Card>
  );
}
