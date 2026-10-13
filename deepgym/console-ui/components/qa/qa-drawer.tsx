'use client';

import { useState } from 'react';
import { TaskRows } from '@/components/primitives/task-rows';
import { Drawer } from '@/components/ui/overlay';
import { VerdictBanner } from '@/components/ui/verdict-banner';
import { timeAgo } from '@/lib/format';
import type { QAReport } from '@/lib/types';
import { failuresFirst, qaStatus, verdictLine } from '@/lib/verdict';

/** One environment's QA checks: the verdict on top, failed checks first, each with its details inline. */
export function QADrawer({ report, onClose }: { report: QAReport | null; onClose: () => void }) {
  // Keep the last report while the drawer animates closed.
  const [shown, setShown] = useState(report);
  const changed = report?.environment !== shown?.environment || report?.created_at !== shown?.created_at;
  if (report !== null && changed) setShown(report);

  const skipped = shown !== null && qaStatus(shown) === 'skip';
  const failed = shown?.checks.filter((check) => !check.passed).length ?? 0;

  return (
    <Drawer
      isOpen={report !== null}
      onClose={onClose}
      title={shown?.environment ?? ''}
      aria-label="QA report"
    >
      {shown && (
        <div className="flex flex-col gap-4">
          <VerdictBanner passed={skipped ? null : shown.passed} meta={timeAgo(shown.created_at)}>
            {skipped
              ? 'QA skipped: this environment cannot be checked'
              : verdictLine('QA', shown.passed, failed, shown.checks.length, 'check')}
          </VerdictBanner>
          <TaskRows
            rows={failuresFirst(
              shown.checks.map((check) => ({
                key: check.name,
                label: check.name,
                subline: check.details || undefined,
                status: check.passed ? 'done' : 'failed',
              })),
            )}
          />
        </div>
      )}
    </Drawer>
  );
}
