import type { Delta } from '@/lib/stats';
import { cx } from '@/utils/cx';

const TONE: Record<Delta['tone'], string> = {
  good: 'text-status-lime-text',
  bad: 'text-status-rose-text',
  neutral: 'text-text-secondary',
};

interface StatProps {
  label: string;
  value: string;
  /** Change vs the previous window, coloured by whether it is an improvement. */
  delta?: Delta | null;
  /** Secondary line, after the delta when there is one (e.g. "vs prev 7d"). */
  hint?: string;
}

/** Headline number tile with an optional delta. */
export function Stat({ label, value, delta, hint }: StatProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-2xl border border-border-button-default bg-background-secondary-default/70 px-4 py-3 shadow-card">
      <span className="text-mono-label text-text-tertiary">{label}</span>
      <span className="font-mono text-title-2-semibold text-text-primary tabular-nums">{value}</span>
      {(delta || hint) && (
        <span className="truncate text-caption-1-medium text-text-secondary">
          {delta && <span className={cx('font-mono tabular-nums', TONE[delta.tone])}>{delta.change}</span>}
          {delta && hint && ' '}
          {hint}
        </span>
      )}
    </div>
  );
}
