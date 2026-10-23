'use client';

import { RiPlayCircleLine } from '@remixicon/react';
import type { ColumnDef } from '@tanstack/react-table';
import { Chip } from '@/components/base/badges/chip';
import { DataTable, type Paging } from '@/components/ui/data-table';
import { EmptyState } from '@/components/ui/empty-state';
import { ScoreBar } from '@/components/ui/score-bar';
import { PassBadge, RunStatusBadge } from '@/components/ui/status-badge';
import { formatDuration, timeAgo } from '@/lib/format';
import type { RunSummary } from '@/lib/types';

const columns: ColumnDef<RunSummary>[] = [
  {
    id: 'environment',
    accessorKey: 'environment',
    header: 'Environment',
    cell: ({ row }) => (
      <span className="font-mono text-body-2-medium text-text-primary">{row.original.environment}</span>
    ),
  },
  {
    id: 'status',
    accessorKey: 'status',
    header: 'Status',
    cell: ({ row }) => <RunStatusBadge status={row.original.status} />,
  },
  {
    id: 'score',
    accessorKey: 'score',
    header: 'Score',
    cell: ({ row }) => <ScoreBar score={row.original.score} />,
  },
  {
    id: 'passed',
    accessorKey: 'passed',
    header: 'Result',
    cell: ({ row }) => <PassBadge passed={row.original.passed} />,
  },
  {
    id: 'error_type',
    accessorKey: 'error_type',
    header: 'Error',
    cell: ({ row }) => (
      <span className="font-mono text-caption-1-medium text-text-error-primary">
        {row.original.error_type ?? ''}
      </span>
    ),
  },
  {
    id: 'provider',
    accessorKey: 'provider',
    header: 'Provider',
    meta: { className: 'font-mono text-text-secondary' },
  },
  {
    id: 'reference',
    accessorKey: 'reference',
    header: 'Solution',
    cell: ({ row }) => (
      <Chip variant="caption" color="neutral">
        {row.original.reference ? 'reference' : 'custom'}
      </Chip>
    ),
  },
  {
    id: 'duration',
    accessorKey: 'duration_ms',
    header: 'Duration',
    cell: ({ row }) => formatDuration(row.original.duration_ms),
    meta: { align: 'end', className: 'font-mono text-text-secondary' },
  },
  {
    id: 'created',
    accessorKey: 'created_at',
    header: 'Started',
    cell: ({ row }) => timeAgo(row.original.created_at),
    meta: { align: 'end', className: 'whitespace-nowrap text-text-secondary' },
  },
];

interface RunsTableProps {
  runs: RunSummary[];
  paging: Paging;
  loading: boolean;
  error?: string;
  onOpen: (run: RunSummary) => void;
}

/** One server page of runs, newest first; Enter or click opens the run. */
export function RunsTable({ runs, paging, loading, error, onOpen }: RunsTableProps) {
  return (
    <DataTable
      label="Runs"
      data={runs}
      columns={columns}
      onRowAction={onOpen}
      serverPaging={paging}
      empty={
        <EmptyState
          icon={RiPlayCircleLine}
          title={error ? 'Could not load runs' : loading ? 'Loading runs' : 'No runs yet'}
        >
          {error ??
            (loading ? null : 'Start one with New run (press N), or run a reference from an environment.')}
        </EmptyState>
      }
    />
  );
}
