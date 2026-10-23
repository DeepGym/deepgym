'use client';

import { RiArrowLeftLine, RiPlayCircleLine, RiRestartLine, RiStackLine } from '@remixicon/react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Button } from '@/components/base/buttons/button';
import { type TaskRow, TaskRows } from '@/components/primitives/task-rows';
import { ActionError } from '@/components/ui/action-error';
import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageHeader } from '@/components/ui/page-header';
import { VerdictBanner } from '@/components/ui/verdict-banner';
import { api, errorText, useApi } from '@/lib/api';
import { formatScore, timeAgo } from '@/lib/format';
import type { RunDetail, RunPage, RunSummary } from '@/lib/types';
import { useQueryParam } from '@/lib/use-query-param';
import { failuresFirst, isActive, runTaskStatus, verdictLine } from '@/lib/verdict';
import { RunProgress } from './run-progress';
import { RunSource } from './run-source';
import { RunSummary as RunSummaryStrip } from './run-summary';
import { ScoreBreakdown } from './score-breakdown';

function GroupRuns({ run }: { run: RunDetail }) {
  const { data: page } = useApi<RunPage>(`/runs?group_id=${run.group_id}`, 2500, ({ runs }) =>
    runs.some(isActive),
  );
  const data = page?.runs;
  if (!data || data.length < 2) return null;
  const running = data.filter(isActive).length;
  const failed = data.filter((item) => !isActive(item) && !item.passed).length;
  const rows: TaskRow[] = data.map((item) => ({
    key: item.id,
    label: <span className={item.id === run.id ? 'text-accent-400' : undefined}>{item.environment}</span>,
    subline: item.error_type ?? undefined,
    meta: formatScore(item.score),
    status: runTaskStatus(item),
    href: `/runs/detail/?id=${item.id}`,
  }));
  return (
    <Card title={`Eval ${run.group_id} - ${data.length} runs`}>
      <div className="flex flex-col gap-3">
        <VerdictBanner passed={running ? null : failed === 0}>
          {running
            ? `Eval running: ${data.length - running} of ${data.length} runs done`
            : verdictLine('Eval', failed === 0, failed, data.length, 'run')}
        </VerdictBanner>
        <TaskRows rows={failuresFirst(rows)} />
      </div>
    </Card>
  );
}

/** One run: headline facts, per-case results, solution and verifier output. Polls until it finishes. */
export function RunDetailPage() {
  const router = useRouter();
  const [id] = useQueryParam('id');
  const { data: run, error } = useApi<RunDetail>(id ? `/runs/${id}` : null, 1500, isActive);
  const [rerunning, setRerunning] = useState(false);
  // Kept with the run it was for, since the page stays mounted when ?id= changes.
  const [rerunError, setRerunError] = useState<{ id: string; message: string }>();

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
    setRerunError(undefined);
    try {
      const body = {
        environments: [run.environment],
        mode: run.mode,
        solution: run.reference ? null : run.solution,
      };
      const [created] = await api<RunSummary[]>('/runs', { method: 'POST', body: JSON.stringify(body) });
      router.push(`/runs/detail/?id=${created.id}`);
    } catch (caught) {
      setRerunError({ id: run.id, message: errorText(caught) });
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
              onClick={() => router.push(`/environments/?env=${encodeURIComponent(run.environment)}`)}
            >
              Environment
            </Button>
            <Button
              size="small"
              leadingIcon={RiRestartLine}
              disabled={rerunning || isActive(run)}
              onClick={rerun}
            >
              Run again
            </Button>
          </>
        }
      />
      {rerunError?.id === run.id && (
        <ActionError className="mb-4">Could not run again: {rerunError.message}</ActionError>
      )}
      <RunSummaryStrip run={run} />
      {isActive(run) && <RunProgress run={run} />}
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
