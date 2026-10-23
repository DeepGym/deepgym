import { Chip, type ChipProps } from '@/components/base/badges/chip';
import { formatScore } from '@/lib/format';
import type { EnvKind, EpisodeSummary, RunStatus } from '@/lib/types';
import { scoreTone, type Tone } from '@/lib/verdict';
import { cx } from '@/utils/cx';

type ChipColor = NonNullable<ChipProps['color']>;

const RUN_STATUS: Record<RunStatus, { color: ChipColor; dot: string; pulse?: boolean }> = {
  queued: { color: 'neutral', dot: 'bg-foreground-icon-secondary' },
  running: { color: 'blue', dot: 'bg-accent-400', pulse: true },
  completed: { color: 'lime', dot: 'bg-status-lime-text' },
  failed: { color: 'rose', dot: 'bg-status-rose-text' },
};

/** Pill for a run's lifecycle status, with a pulsing dot while it runs. */
export function RunStatusBadge({ status }: { status: RunStatus }) {
  const { color, dot, pulse } = RUN_STATUS[status];
  return (
    <Chip variant="caption" color={color} className="gap-1.5">
      <span aria-hidden className={cx('size-1.5 rounded-full', dot, pulse && 'animate-pulse')} />
      {status}
    </Chip>
  );
}

/** Pass/fail pill; a dash while unknown. */
export function PassBadge({ passed }: { passed: boolean | null }) {
  if (passed === null) return <span className="text-text-tertiary">-</span>;
  return (
    <Chip variant="caption" color={passed ? 'lime' : 'rose'}>
      {passed ? 'pass' : 'fail'}
    </Chip>
  );
}

/**
 * Pass/fail pill, or "no verdict" and its error type when the verifier or sandbox failed instead
 * of judging, so that 0 is never shown as a fail.
 */
export function VerdictBadge({
  verdict,
}: {
  verdict: Pick<EpisodeSummary, 'passed' | 'has_verdict' | 'error_type'>;
}) {
  if (verdict.has_verdict !== false) return <PassBadge passed={verdict.passed} />;
  return (
    <span className="inline-flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
      <Chip variant="caption" color="yellow">
        no verdict
      </Chip>
      <span className="font-mono text-caption-1-medium text-text-error-primary">{verdict.error_type}</span>
    </span>
  );
}

const TONE_COLOR: Record<Tone, ChipColor> = { good: 'lime', warn: 'yellow', bad: 'rose' };

/** A score in [0, 1] as a pill coloured by its band (green from 0.8, amber from 0.5, red below). */
export function ScoreChip({ score }: { score: number }) {
  return (
    <Chip variant="caption" color={TONE_COLOR[scoreTone(score)]} className="font-mono tabular-nums">
      {formatScore(score)}
    </Chip>
  );
}

const KIND_COLOR: Record<EnvKind, ChipColor> = {
  'single-turn': 'neutral',
  'multi-turn': 'cyan',
  'computer-use': 'blue',
};

/** Chip for an environment's interaction kind. */
export function KindBadge({ kind }: { kind: EnvKind }) {
  return (
    <Chip variant="caption" color={KIND_COLOR[kind]}>
      {kind}
    </Chip>
  );
}
