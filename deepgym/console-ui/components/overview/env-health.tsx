import { RiStackLine } from '@remixicon/react';
import Link from 'next/link';
import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { formatDuration, formatPercent } from '@/lib/format';
import { passRate, worstFirst } from '@/lib/stats';
import type { EnvironmentStats } from '@/lib/types';
import { scoreTone, type Tone } from '@/lib/verdict';
import { cx } from '@/utils/cx';

const BORDER: Record<Tone, string> = {
  good: 'border-status-lime-text/50',
  warn: 'border-status-yellow-text/50',
  bad: 'border-status-rose-text/50',
};

/** One environment's window at a glance; the border is tinted by its pass rate. Opens its runs. */
function HealthCard({ env }: { env: EnvironmentStats }) {
  const rate = passRate(env);
  const cells = [
    { label: 'Runs', value: String(env.runs) },
    { label: 'Pass', value: formatPercent(rate) },
    { label: 'No verdict', value: String(env.no_verdict_count), alert: env.no_verdict_count > 0 },
    { label: 'Avg time', value: formatDuration(env.mean_duration_ms) },
    { label: 'Errors', value: String(env.errors), alert: env.errors > 0 },
  ];
  return (
    <Link
      href={`/runs/?environment=${encodeURIComponent(env.environment)}`}
      className={cx(
        'flex min-w-0 flex-col gap-3 rounded-2xl border bg-background-primary-default px-4 py-3 shadow-card outline-none transition-colors hover:bg-background-secondary-hover focus-visible:ring-2 focus-visible:ring-border-focus-ring',
        rate === null ? 'border-border-button-default' : BORDER[scoreTone(rate)],
      )}
    >
      <span className="truncate font-mono text-body-2-medium text-text-primary">{env.environment}</span>
      <dl className="grid grid-cols-5 gap-2">
        {cells.map((cell) => (
          <div key={cell.label} className="flex min-w-0 flex-col gap-0.5">
            <dt className="text-mono-label truncate text-text-tertiary">{cell.label}</dt>
            <dd
              className={cx(
                'truncate font-mono text-body-2-medium tabular-nums',
                cell.alert ? 'text-text-error-primary' : 'text-text-primary',
              )}
            >
              {cell.value}
            </dd>
          </div>
        ))}
      </dl>
    </Link>
  );
}

interface EnvHealthProps {
  /** Per-environment totals of the window; undefined while loading. */
  environments: EnvironmentStats[] | undefined;
  error?: string;
}

/** A card per environment run in the window, lowest pass rate first. */
export function EnvHealth({ environments, error }: EnvHealthProps) {
  return (
    <Card title="Environment health">
      {environments?.length ? (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {worstFirst(environments).map((env) => (
            <HealthCard key={env.environment} env={env} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={RiStackLine}
          title={error ? 'Could not load stats' : environments ? 'No runs in this window' : 'Loading stats'}
        >
          {error ?? (environments ? 'Start one with New run, or pick a longer range.' : null)}
        </EmptyState>
      )}
    </Card>
  );
}
