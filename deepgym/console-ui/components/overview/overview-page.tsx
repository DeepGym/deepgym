'use client';

import { RiAddLine, RiPulseLine } from '@remixicon/react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/base/buttons/button';
import { TaskRows } from '@/components/primitives/task-rows';
import { KpiTiles } from '@/components/runs/run-stats';
import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageHeader } from '@/components/ui/page-header';
import { RangeControl, useRange } from '@/components/ui/range-control';
import { useApi } from '@/lib/api';
import { timeAgo } from '@/lib/format';
import { RANGES } from '@/lib/stats';
import type { QAReport, RunStats, RunSummary } from '@/lib/types';
import { recentActivity } from '@/lib/verdict';
import { EnvHealth } from './env-health';

/** Home: KPIs for the chosen window (?range=) vs the one before, environment health, and recent activity. */
export function OverviewPage() {
  const router = useRouter();
  const [range, setRange] = useRange();
  const { data: stats, error } = useApi<RunStats>(`/runs/stats?hours=${RANGES[range]}`, 5000);
  const { data: runs } = useApi<RunSummary[]>('/runs?limit=10', 5000);
  const { data: reports } = useApi<QAReport[]>('/qa', 15_000);
  const activity = recentActivity(runs ?? [], reports ?? [], 10);

  return (
    <>
      <PageHeader
        title="Overview"
        description="Runs, pass rates and errors across environments, compared with the window before."
        actions={
          <>
            <RangeControl value={range} onChange={setRange} />
            <Button leadingIcon={RiAddLine} onClick={() => router.push('/runs/?new=1')}>
              New run
            </Button>
          </>
        }
      />
      <KpiTiles stats={stats} range={range} />
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <EnvHealth environments={stats?.environments} error={error} />
        <Card
          title="Recent activity"
          className="self-start"
          actions={
            <Link
              href="/runs/"
              className="rounded-sm text-caption-1-medium text-accent-400 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-border-focus-ring"
            >
              All runs
            </Link>
          }
        >
          {activity.length ? (
            <TaskRows rows={activity.map((item) => ({ ...item, meta: timeAgo(item.at) }))} />
          ) : (
            <EmptyState icon={RiPulseLine} title={runs && reports ? 'No activity yet' : 'Loading activity'}>
              {runs && reports ? 'Runs and QA verdicts show up here as they happen.' : null}
            </EmptyState>
          )}
        </Card>
      </div>
    </>
  );
}
