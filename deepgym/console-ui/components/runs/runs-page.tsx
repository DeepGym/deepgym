'use client';

import { RiAddLine, RiCloseLine } from '@remixicon/react';
import { useRouter } from 'next/navigation';
import { Chip } from '@/components/base/badges/chip';
import { Button } from '@/components/base/buttons/button';
import {
  SegmentedControl,
  SegmentedControlItem,
} from '@/components/base/segmented-control/segmented-control';
import { Card } from '@/components/ui/card';
import { FilterSelect } from '@/components/ui/filter-select';
import { PageHeader } from '@/components/ui/page-header';
import { RangeControl, useRange } from '@/components/ui/range-control';
import { Stat } from '@/components/ui/stat';
import { useApi } from '@/lib/api';
import { RANGES } from '@/lib/stats';
import { RUN_MODES, type RunStats, type RunSummary } from '@/lib/types';
import { useHotkey } from '@/lib/use-hotkey';
import { useQueryParam } from '@/lib/use-query-param';
import { isActive } from '@/lib/verdict';
import { NewRunDialog } from './new-run-dialog';
import { KpiTiles, ScoreDistribution } from './run-stats';
import { RunsTable } from './runs-table';

const STATUSES = ['all', 'queued', 'running', 'completed', 'failed'] as const;
const PROVIDERS = RUN_MODES.filter((mode) => mode !== 'auto');

function FilterChip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <Chip variant="caption" color="blue" className="gap-1 ps-2 pe-1">
      <span className="font-mono">{label}</span>
      <button
        type="button"
        aria-label={`Clear ${label}`}
        onClick={onClear}
        className="flex cursor-pointer rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring"
      >
        <RiCloseLine aria-hidden className="size-3.5" />
      </button>
    </Chip>
  );
}

/**
 * Stats for the chosen window (?range=) with deltas, and the runs table with status, provider,
 * environment and eval-group filters, plus the New run dialog (?new=).
 */
export function RunsPage() {
  const router = useRouter();
  const [status, setStatus] = useQueryParam('status');
  const [provider, setProvider] = useQueryParam('provider');
  const [environment, setEnvironment] = useQueryParam('environment');
  const [group, setGroup] = useQueryParam('group');
  const [newRun, setNewRun] = useQueryParam('new');
  useHotkey('n', () => setNewRun('1'), newRun === null);

  const [range, setRange] = useRange();

  const filters = new URLSearchParams();
  if (status) filters.set('status', status);
  if (provider) filters.set('provider', provider);
  if (environment) filters.set('environment', environment);
  if (group) filters.set('group_id', group);
  const { data, error } = useApi<RunSummary[]>(`/runs?${filters}&limit=500`, 2500);
  // An eval group is one batch, so its stats cover all of it rather than a time window.
  const { data: stats } = useApi<RunStats>(
    `/runs/stats?${filters}${group ? '' : `&hours=${RANGES[range]}`}`,
    5000,
  );
  const runs = data ?? [];

  return (
    <>
      <PageHeader
        title="Runs"
        description="Score a solution, or a reference, against one environment or a small eval. Updates live."
        actions={
          <>
            {!group && <RangeControl value={range} onChange={setRange} />}
            <Button leadingIcon={RiAddLine} onClick={() => setNewRun('1')}>
              New run
            </Button>
          </>
        }
      />
      <KpiTiles stats={stats} range={group ? null : range}>
        <Stat label="In flight" value={String(runs.filter(isActive).length)} hint="queued or running" />
      </KpiTiles>
      <div className="mt-3">
        <ScoreDistribution bins={stats?.score_bins} />
      </div>
      <Card flush className="mt-4">
        <div className="flex flex-wrap items-center gap-2 p-3">
          <div className="scrollbar-thin max-w-full overflow-x-auto">
            <SegmentedControl
              aria-label="Status"
              selectedKeys={[status ?? 'all']}
              onSelectionChange={(keys) => {
                const next = [...keys][0];
                setStatus(next === 'all' ? null : String(next));
              }}
            >
              {STATUSES.map((value) => (
                <SegmentedControlItem key={value} id={value}>
                  {value}
                </SegmentedControlItem>
              ))}
            </SegmentedControl>
          </div>
          <FilterSelect label="providers" value={provider} options={[...PROVIDERS]} onChange={setProvider} />
          {environment && <FilterChip label={environment} onClear={() => setEnvironment(null)} />}
          {group && <FilterChip label={`eval ${group}`} onClear={() => setGroup(null)} />}
          <span className="ms-auto font-mono text-caption-1-medium text-text-tertiary">
            {runs.length} runs
          </span>
        </div>
        <RunsTable
          runs={runs}
          loading={!data && !error}
          error={error}
          onOpen={(run) => router.push(`/runs/detail/?id=${run.id}`)}
        />
      </Card>
      <NewRunDialog
        open={newRun !== null}
        initialEnv={newRun && newRun !== '1' ? newRun : null}
        onClose={() => setNewRun(null)}
        onCreated={(created) =>
          created.length === 1
            ? router.push(`/runs/detail/?id=${created[0].id}`)
            : router.replace(`/runs/?group=${created[0].group_id}`)
        }
      />
    </>
  );
}
