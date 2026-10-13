'use client';

import type { ReactNode } from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from 'recharts';
import { Stat } from '@/components/ui/stat';
import { formatPercent, formatScore } from '@/lib/format';
import { type Better, type DeltaUnit, delta, passRate, type Range } from '@/lib/stats';
import type { RunStats, RunTotals } from '@/lib/types';

interface Kpi {
  label: string;
  value: (totals: RunTotals) => number | null;
  format: (value: number | null) => string;
  unit: DeltaUnit;
  better: Better;
  /** Hint when there is no delta to show. */
  hint: (totals: RunTotals) => string;
}

const count = (value: number | null) => String(value ?? 0);

const KPIS: Kpi[] = [
  {
    label: 'Runs',
    value: (t) => t.runs,
    format: count,
    unit: 'count',
    better: 'neither',
    hint: (t) => `${t.finished} finished`,
  },
  {
    label: 'Pass rate',
    value: passRate,
    format: formatPercent,
    unit: 'rate',
    better: 'higher',
    hint: (t) => `${t.passed} passed`,
  },
  {
    label: 'Avg score',
    value: (t) => t.mean_score,
    format: formatScore,
    unit: 'score',
    better: 'higher',
    hint: () => 'finished runs',
  },
  {
    label: 'Errors',
    value: (t) => t.errors,
    format: count,
    unit: 'count',
    better: 'lower',
    hint: () => 'runs with an error type',
  },
];

interface KpiTilesProps {
  stats: RunStats | undefined;
  /** The window `stats` covers, for "vs prev 7d"; null when it has no previous window. */
  range: Range | null;
  /** Extra tiles after the KPIs. */
  children?: ReactNode;
}

/** Runs, pass rate, avg score and errors, each with its change vs the previous window. */
export function KpiTiles({ stats, range, children }: KpiTilesProps) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-[repeat(auto-fit,minmax(10rem,1fr))]">
      {KPIS.map((kpi) => {
        if (!stats) return <Stat key={kpi.label} label={kpi.label} value="-" />;
        const { current, previous } = stats;
        const change = previous && delta(kpi.value(current), kpi.value(previous), kpi.unit, kpi.better);
        return (
          <Stat
            key={kpi.label}
            label={kpi.label}
            value={kpi.format(kpi.value(current))}
            delta={change}
            hint={change ? `vs prev ${range}` : kpi.hint(current)}
          />
        );
      })}
      {children}
    </div>
  );
}

/** Small translucent panel with a mono label, sized to sit next to another. */
function Panel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col rounded-2xl border border-border-button-default bg-background-secondary-default/70 px-4 pt-3 pb-1 shadow-card">
      <h2 className="text-mono-label text-text-tertiary">{label}</h2>
      {children}
    </section>
  );
}

/** Finished runs per score tenth (0.0-0.1 ... 0.9-1.0). */
export function ScoreDistribution({ bins }: { bins: number[] | undefined }) {
  const data = (bins ?? Array<number>(10).fill(0)).map((runs, index) => ({
    bin: (index / 10).toFixed(1),
    runs,
  }));
  return (
    <Panel label="Score distribution">
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 0, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="bin"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 10, fill: 'var(--color-text-tertiary)' }}
              interval={1}
            />
            <Tooltip
              cursor={{ fill: 'var(--color-background-secondary-hover)' }}
              contentStyle={{
                background: 'var(--color-background-primary-default)',
                border: '1px solid var(--color-border-button-default)',
                borderRadius: 8,
                fontSize: 12,
              }}
              labelFormatter={(bin) => `score ${bin}+`}
            />
            <Bar dataKey="runs" fill="var(--color-chart-1)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

const TOP_ERROR_TYPES = 6;

/** The most common error types of the window, each with its count and share of the errored runs. */
export function FailureModes({ errorTypes }: { errorTypes: RunStats['error_types'] | undefined }) {
  const all = errorTypes ?? [];
  const total = all.reduce((sum, item) => sum + item.count, 0);
  return (
    <Panel label="Failure modes">
      {total === 0 ? (
        <p className="flex flex-1 items-center justify-center py-6 text-body-2-regular text-text-tertiary">
          {errorTypes ? 'No errors in this window' : 'Loading'}
        </p>
      ) : (
        <ul aria-label="Error types" className="flex flex-col gap-2.5 pt-3 pb-3">
          {all.slice(0, TOP_ERROR_TYPES).map((item) => {
            const share = item.count / total;
            return (
              <li
                key={item.error_type}
                className="grid grid-cols-[minmax(0,11rem)_minmax(0,1fr)_5.5rem] items-center gap-3"
              >
                <span
                  title={item.error_type}
                  className="truncate font-mono text-caption-1-medium text-text-primary"
                >
                  {item.error_type}
                </span>
                <div aria-hidden className="h-2 overflow-hidden rounded-full bg-background-tertiary-default">
                  <div
                    className="h-full rounded-full bg-status-rose-text"
                    style={{ width: `${share * 100}%` }}
                  />
                </div>
                <span className="text-end font-mono text-caption-1-medium text-text-primary tabular-nums">
                  {item.count} <span className="text-text-tertiary">{formatPercent(share)}</span>
                </span>
              </li>
            );
          })}
          {all.length > TOP_ERROR_TYPES && (
            <li className="text-caption-1-medium text-text-tertiary">
              +{all.length - TOP_ERROR_TYPES} more error types
            </li>
          )}
        </ul>
      )}
    </Panel>
  );
}
