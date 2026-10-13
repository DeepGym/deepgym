'use client';

import { RiArrowLeftLine, RiPlayCircleLine, RiRestartLine, RiStackLine } from '@remixicon/react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Button } from '@/components/base/buttons/button';
import { type TaskRow, TaskRows } from '@/components/primitives/task-rows';
import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageHeader } from '@/components/ui/page-header';
import { api, useApi } from '@/lib/api';
import { formatScore, timeAgo } from '@/lib/format';
import type { RunDetail, RunSummary } from '@/lib/types';
import { useQueryParam } from '@/lib/use-query-param';
import { RunSource } from './run-source';
import { RunSummary as RunSummaryStrip } from './run-summary';
import { ScoreBreakdown } from './score-breakdown';

const active = (run: { status: string }) => run.status === 'queued' || run.status === 'running';

const taskStatus = (run: RunSummary): TaskRow['status'] =>
  active(run) ? (run.status as 'queued' | 'running') : run.passed ? 'done' : 'failed';

function GroupRuns({ run }: { run: RunDetail }) {
  const { data } = useApi<RunSummary[]>(`/runs?group_id=${run.group_id}`, 2500, (runs) => runs.some(active));
  if (!data || data.length < 2) return null;
  const rows: TaskRow[] = data.map((item) => ({
    key: item.id,
    label: <span className={item.id === run.id ? 'text-accent-400' : undefined}>{item.environment}</span>,
    meta: formatScore(item.score),
    status: taskStatus(item),
    href: `/runs/detail/?id=${item.id}`,
  }));
  return (
    <Card title={`Eval ${run.group_id} - ${data.length} runs`}>
      <TaskRows rows={rows} />
    </Card>
  );
}

/** One run: headline facts, per-case results, solution and verifier output. Polls until it finishes. */
export function RunDetailPage() {
  const router = useRouter();
  const [id] = useQueryParam('id');
  const { data: run, error } = useApi<RunDetail>(id ? `/runs/${id}` : null, 1500, active);
  const [rerunning, setRerunning] = useState(false);

  if (!id || error) {
    return (
      <EmptyState icon={RiPlayCircleLine} title={id ? 'Run not found' : 'No run selected'}>
        {error ?? <Link href="/runs/">Back to runs</Link>}
      </EmptyState>
    );
  }
  if (!run) return <p className="py-10 text-body-regular text-text-tertiary">Loading run {id}</p>;

  const rerun = async () => {
    setRerunning(true);
    try {
      const body = {
        environments: [run.environment],
        mode: run.mode,
        solution: run.reference ? null : run.solution,
      };
      const [created] = await api<RunSummary[]>('/runs', { method: 'POST', body: JSON.stringify(body) });
      router.push(`/runs/detail/?id=${created.id}`);
    } finally {
      setRerunning(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow={
          <Link href="/runs/" className="inline-flex items-center gap-1 hover:text-text-primary">
            <RiArrowLeftLine aria-hidden className="size-3" /> Runs / {run.id}
          </Link>
        }
        title={run.environment}
        description={`${run.reference ? 'Reference' : 'Custom'} solution, started ${timeAgo(run.created_at)}`}
        actions={
          <>
            <Button
              size="small"
              variant="secondary"
              leadingIcon={RiStackLine}
              onClick={() => router.push(`/?env=${encodeURIComponent(run.environment)}`)}
            >
              Environment
            </Button>
            <Button
              size="small"
              leadingIcon={RiRestartLine}
              disabled={rerunning || active(run)}
              onClick={rerun}
            >
              Run again
            </Button>
          </>
        }
      />
      <RunSummaryStrip run={run} />
      {run.error && (
        <pre className="mt-4 overflow-x-auto rounded-2xl bg-background-tertiary-error px-4 py-3 font-mono text-body-2-regular whitespace-pre-wrap text-text-error-primary">
          {run.error}
        </pre>
      )}
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="flex min-w-0 flex-col gap-4">
          {run.result && <ScoreBreakdown result={run.result} />}
          <RunSource run={run} />
        </div>
        <div className="flex min-w-0 flex-col gap-4">
          <GroupRuns run={run} />
          <Card title="Identifiers">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 font-mono text-caption-1-medium">
              <dt className="text-text-tertiary">run</dt>
              <dd className="truncate text-text-primary">{run.id}</dd>
              <dt className="text-text-tertiary">eval</dt>
              <dd className="truncate text-text-primary">{run.group_id}</dd>
              <dt className="text-text-tertiary">sandbox</dt>
              <dd className="truncate text-text-primary">{run.result?.sandbox_id ?? '-'}</dd>
            </dl>
          </Card>
        </div>
      </div>
    </>
  );
}
