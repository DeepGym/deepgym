'use client';

import type { ColumnDef } from '@tanstack/react-table';
import { DataTable } from '@/components/ui/data-table';
import { PassBadge } from '@/components/ui/status-badge';
import { formatDuration, formatScore } from '@/lib/format';
import type { CaseResult } from '@/lib/types';
import { caseReason } from '@/lib/verdict';

type CaseRow = CaseResult & { name: string };

const clip = 'block max-w-64 truncate font-mono text-caption-1-medium';

const columns: ColumnDef<CaseRow>[] = [
  { id: 'name', accessorKey: 'name', header: 'Case', meta: { className: 'font-mono text-text-primary' } },
  {
    id: 'passed',
    accessorFn: (row) => Number(row.passed),
    header: 'Verdict',
    cell: ({ row }) => <PassBadge passed={row.original.passed} />,
  },
  {
    id: 'input',
    accessorKey: 'input_summary',
    header: 'Input',
    enableSorting: false,
    cell: ({ row }) => (
      <span title={row.original.input_summary} className={`${clip} text-text-secondary`}>
        {row.original.input_summary}
      </span>
    ),
  },
  {
    id: 'details',
    accessorFn: caseReason,
    header: 'Details',
    enableSorting: false,
    cell: ({ row }) => {
      const reason = caseReason(row.original);
      return (
        <span
          title={reason}
          className={`${clip} max-w-96 ${row.original.passed ? 'text-text-secondary' : 'text-text-primary'}`}
        >
          {reason}
        </span>
      );
    },
  },
  {
    id: 'score',
    accessorKey: 'score',
    header: 'Score',
    cell: ({ row }) => formatScore(row.original.score),
    meta: { align: 'end', className: 'font-mono' },
  },
  {
    id: 'time',
    accessorKey: 'execution_time_ms',
    header: 'Time',
    cell: ({ row }) => formatDuration(row.original.execution_time_ms),
    meta: { align: 'end', className: 'font-mono text-text-secondary' },
  },
];

/** Per-case verdicts with the reason for each; failures sort first. */
export function CasesTable({ cases }: { cases: CaseResult[] }) {
  const rows = cases.map((item, index) => ({
    ...item,
    id: String(index),
    name: item.id || `case ${index + 1}`,
  }));
  return (
    <DataTable
      label="Test cases"
      data={rows}
      columns={columns}
      initialSort={[{ id: 'passed', desc: false }]}
      pageSize={25}
    />
  );
}
