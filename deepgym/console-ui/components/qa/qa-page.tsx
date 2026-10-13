'use client';

import { RiShieldCheckLine } from '@remixicon/react';
import type { ColumnDef } from '@tanstack/react-table';
import { Card } from '@/components/ui/card';
import { DataTable } from '@/components/ui/data-table';
import { EmptyState } from '@/components/ui/empty-state';
import { PageHeader } from '@/components/ui/page-header';
import { useApi } from '@/lib/api';
import { timeAgo } from '@/lib/format';
import type { QAReport } from '@/lib/types';
import { useQueryParam } from '@/lib/use-query-param';
import { qaStatus } from '@/lib/verdict';
import { QADrawer } from './qa-drawer';
import { QAStatusBadge } from './qa-status';

type QARow = QAReport & { id: string };

const STATUS_ORDER = { fail: 0, skip: 1, pass: 2 };

const columns: ColumnDef<QARow>[] = [
  { id: 'id', accessorKey: 'id', header: 'Environment', meta: { className: 'font-mono text-text-primary' } },
  {
    id: 'status',
    accessorFn: (row) => STATUS_ORDER[qaStatus(row)],
    header: 'Result',
    cell: ({ row }) => <QAStatusBadge report={row.original} />,
  },
  {
    id: 'checks',
    accessorFn: (row) => row.checks.filter((check) => check.passed).length / (row.checks.length || 1),
    header: 'Checks passed',
    cell: ({ row }) =>
      `${row.original.checks.filter((check) => check.passed).length} / ${row.original.checks.length}`,
    meta: { className: 'font-mono text-text-secondary' },
  },
  {
    id: 'failed',
    accessorFn: (row) =>
      row.checks
        .filter((check) => !check.passed)
        .map((check) => check.name)
        .join(', '),
    header: 'Failing',
    enableSorting: false,
    meta: { className: 'font-mono text-caption-1-medium text-text-error-primary' },
  },
  {
    id: 'created',
    accessorKey: 'created_at',
    header: 'Checked',
    cell: ({ row }) => timeAgo(row.original.created_at),
    meta: { align: 'end', className: 'whitespace-nowrap text-text-secondary' },
  },
];

/** `deepgym qa` verdicts, one per environment, failures first; a row opens its checks (?env=). */
export function QAPage() {
  const { data, error } = useApi<QAReport[]>('/qa', 15_000);
  const [selected, setSelected] = useQueryParam('env');
  const rows = (data ?? []).map((report) => ({ ...report, id: report.environment }));

  return (
    <>
      <PageHeader
        title="QA"
        description="Whether each environment is fit to train on: reference passes, stubs fail, reward hacks fail."
      />
      <Card flush>
        <DataTable
          label="QA reports"
          data={rows}
          columns={columns}
          onRowAction={(row) => setSelected(row.id)}
          initialSort={[{ id: 'status', desc: false }]}
          empty={
            <EmptyState
              icon={RiShieldCheckLine}
              title={error ? 'Could not load QA reports' : 'No QA reports yet'}
            >
              {error ?? (
                <>
                  Run <code className="font-mono text-text-primary">deepgym qa --all</code>; reports land in{' '}
                  <code className="font-mono text-text-primary">~/.deepgym/qa/</code>.
                </>
              )}
            </EmptyState>
          }
        />
      </Card>
      <QADrawer report={rows.find((row) => row.id === selected) ?? null} onClose={() => setSelected(null)} />
    </>
  );
}
