'use client';

import { RiArrowDownSLine, RiListCheck3 } from '@remixicon/react';
import { useId, useState } from 'react';
import { Chip } from '@/components/base/badges/chip';
import { Card } from '@/components/ui/card';
import { Collapse } from '@/components/ui/collapse';
import { EmptyState } from '@/components/ui/empty-state';
import { ScoreChip } from '@/components/ui/status-badge';
import type { RunResult } from '@/lib/types';
import { cx } from '@/utils/cx';
import { CasesTable } from './cases-table';

const row = 'flex min-h-11 w-full items-center gap-2 border-t border-separator-border px-4 py-2';

const NamePill = ({ name }: { name: string }) => (
  <Chip variant="caption" color="neutral" className="font-mono text-text-primary">
    {name}
  </Chip>
);

/**
 * What the score is made of: each reward component the verifier reported, then its test
 * cases as a row with pass/fail counts that expands into per-case verdicts (open when the run failed).
 * Verifiers report components as plain numbers, without weights or the cases behind them.
 */
export function ScoreBreakdown({ result }: { result: RunResult }) {
  const cases = result.cases ?? [];
  const failed = cases.filter((item) => !item.passed).length;
  const components = Object.entries(result.reward_components ?? {});
  const [open, setOpen] = useState(!result.passed);
  const tableId = useId();

  return (
    <Card title="Score breakdown" flush>
      {components.length === 0 && cases.length === 0 && (
        <EmptyState icon={RiListCheck3} title="This verifier reports no reward components or cases" />
      )}
      {components.length > 0 && (
        <ul aria-label="Reward components">
          {components.map(([name, value]) => (
            <li key={name} className={row}>
              <NamePill name={name} />
              <ScoreChip score={value} />
            </li>
          ))}
        </ul>
      )}
      {cases.length > 0 && (
        <>
          <button
            type="button"
            aria-expanded={open}
            aria-controls={tableId}
            onClick={() => setOpen((value) => !value)}
            className={cx(
              row,
              'cursor-pointer text-start outline-none transition-colors hover:bg-background-secondary-hover focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-border-focus-ring',
            )}
          >
            <NamePill name="cases" />
            <ScoreChip score={cases.reduce((sum, item) => sum + item.score, 0) / cases.length} />
            <span className="ms-auto flex items-center gap-1.5">
              {cases.length > failed && (
                <Chip variant="caption" color="lime">
                  {cases.length - failed} pass
                </Chip>
              )}
              {failed > 0 && (
                <Chip variant="caption" color="rose">
                  {failed} fail
                </Chip>
              )}
              <Chip variant="caption" color="neutral">
                {cases.length} {cases.length === 1 ? 'case' : 'cases'}
              </Chip>
            </span>
            <RiArrowDownSLine
              aria-hidden
              className={cx(
                'size-4 shrink-0 text-foreground-icon-secondary transition-transform',
                open && 'rotate-180',
              )}
            />
          </button>
          <Collapse open={open} id={tableId}>
            <div className="border-t border-separator-border">
              <CasesTable cases={cases} />
            </div>
          </Collapse>
        </>
      )}
    </Card>
  );
}
