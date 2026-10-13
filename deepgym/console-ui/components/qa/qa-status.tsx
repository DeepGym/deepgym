import { Chip } from '@/components/base/badges/chip';
import type { QAReport } from '@/lib/types';

const COLORS = { pass: 'lime', fail: 'rose', inconclusive: 'yellow', skip: 'neutral' } as const;

/** Chip for a report's verdict. */
export function QAStatusBadge({ report }: { report: QAReport }) {
  return (
    <Chip variant="caption" color={COLORS[report.status]}>
      {report.status}
    </Chip>
  );
}
