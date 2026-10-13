import { Chip } from '@/components/base/badges/chip';
import type { QAReport } from '@/lib/types';
import { qaStatus } from '@/lib/verdict';

const COLORS = { pass: 'lime', fail: 'rose', skip: 'neutral' } as const;

/** Chip for a report's verdict. */
export function QAStatusBadge({ report }: { report: QAReport }) {
  const status = qaStatus(report);
  return (
    <Chip variant="caption" color={COLORS[status]}>
      {status}
    </Chip>
  );
}
