'use client';

import { RiFilmLine } from '@remixicon/react';
import type { ColumnDef } from '@tanstack/react-table';
import { CodeBlock } from '@/components/primitives/code-block';
import { Card } from '@/components/ui/card';
import { DataTable } from '@/components/ui/data-table';
import { EmptyState } from '@/components/ui/empty-state';
import { PageHeader } from '@/components/ui/page-header';
import { VerdictBadge } from '@/components/ui/status-badge';
import { useApi } from '@/lib/api';
import { formatScore, timeAgo } from '@/lib/format';
import type { EpisodeDetail, EpisodePage, EpisodeSummary } from '@/lib/types';
import { usePagedApi } from '@/lib/use-paged-api';
import { useQueryParam } from '@/lib/use-query-param';
import { EpisodeReplay } from './episode-replay';

const RECORD_SNIPPET = `from deepgym import DeepGym, load_environment
from deepgym.console import record_episode

dg = DeepGym(mode='daytona')
trajectory, result = dg.run_episode(load_environment('todo_list'), agent)
record_episode(trajectory, result, environment='todo_list')`;

const columns: ColumnDef<EpisodeSummary>[] = [
  {
    id: 'environment',
    accessorKey: 'environment',
    header: 'Environment',
    cell: ({ row }) => (
      <div className="flex min-w-0 flex-col">
        <span className="truncate font-mono text-body-2-medium text-text-primary">
          {row.original.environment || 'unnamed'}
        </span>
        <span className="text-caption-2-medium text-text-tertiary">{timeAgo(row.original.created_at)}</span>
      </div>
    ),
  },
  { id: 'steps', accessorKey: 'steps', header: 'Steps', meta: { align: 'end', className: 'font-mono' } },
  {
    id: 'score',
    accessorKey: 'score',
    header: 'Score',
    cell: ({ row }) => formatScore(row.original.score),
    meta: { align: 'end', className: 'font-mono' },
  },
  {
    id: 'verdict',
    header: '',
    cell: ({ row }) => <VerdictBadge verdict={row.original} />,
  },
];

function Replay({ id }: { id: string }) {
  const { data, error } = useApi<EpisodeDetail>(`/episodes/${id}`);
  if (error)
    return (
      <EmptyState icon={RiFilmLine} title="Could not load episode">
        {error}
      </EmptyState>
    );
  if (!data) return <p className="text-body-regular text-text-tertiary">Loading episode {id}</p>;
  return <EpisodeReplay key={id} episode={data} />;
}

/** Recorded episodes on the left, a server page at a time; the selected one (?id=) replays on the right. */
export function EpisodesPage() {
  const { data, error, paging } = usePagedApi<EpisodePage>('/episodes', 10_000);
  const [selected, setSelected] = useQueryParam('id');

  return (
    <>
      <PageHeader
        title="Episodes"
        description="Recorded multi-turn and computer-use trajectories, replayed step by step."
      />
      {data?.total === 0 ? (
        <Card>
          <EmptyState icon={RiFilmLine} title="No episodes recorded yet">
            Store a finished episode from your agent loop and it shows up here:
          </EmptyState>
          <CodeBlock code={RECORD_SNIPPET} filename="record.py" className="mx-auto max-w-2xl" />
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[20rem_minmax(0,1fr)]">
          <Card title="Recorded" flush className="self-start">
            <DataTable
              label="Episodes"
              data={data?.episodes ?? []}
              columns={columns}
              onRowAction={(episode) => setSelected(episode.id)}
              serverPaging={paging}
              empty={
                <EmptyState icon={RiFilmLine} title={error ? 'Could not load episodes' : 'Loading episodes'}>
                  {error}
                </EmptyState>
              }
            />
          </Card>
          <div className="min-w-0">
            {selected ? (
              <Replay id={selected} />
            ) : (
              <EmptyState icon={RiFilmLine} title="Pick an episode to replay">
                Arrow keys step through it once it is open.
              </EmptyState>
            )}
          </div>
        </div>
      )}
    </>
  );
}
