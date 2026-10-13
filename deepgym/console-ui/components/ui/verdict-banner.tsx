import { RiCheckboxCircleFill, RiErrorWarningFill, RiTimeLine } from '@remixicon/react';
import type { ReactNode } from 'react';
import { cx } from '@/utils/cx';

interface VerdictBannerProps {
  /** Green when true, red when false, neutral while there is no verdict yet (or it was skipped). */
  passed: boolean | null;
  children: ReactNode;
  /** Right-aligned note, e.g. when the verdict was reached. */
  meta?: string;
}

/** One-line verdict above a list of checks, e.g. "QA failed: 2 of 6 checks". */
export function VerdictBanner({ passed, children, meta }: VerdictBannerProps) {
  const Icon = passed === null ? RiTimeLine : passed ? RiCheckboxCircleFill : RiErrorWarningFill;
  return (
    <div
      role="status"
      className={cx(
        'flex items-center gap-2 rounded-2lg px-3 py-2 text-body-2-medium',
        passed === null && 'bg-background-tertiary-default text-text-secondary',
        passed === true && 'bg-status-lime-background text-status-lime-text',
        passed === false && 'bg-status-rose-background text-status-rose-text',
      )}
    >
      <Icon aria-hidden className="size-4 shrink-0" />
      <span className="min-w-0 flex-1">{children}</span>
      {meta && <span className="shrink-0 text-caption-1-medium opacity-80">{meta}</span>}
    </div>
  );
}
