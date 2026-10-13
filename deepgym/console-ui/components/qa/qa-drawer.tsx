'use client';

import { useState } from 'react';
import { TaskRows } from '@/components/primitives/task-rows';
import { Drawer } from '@/components/ui/overlay';
import { VerdictBanner } from '@/components/ui/verdict-banner';
import { timeAgo } from '@/lib/format';
import type { QAReport } from '@/lib/types';
import { failuresFirst, QA_TASK_STATUS, verdictLine } from '@/lib/verdict';

/** The banner's headline: skipped and inconclusive reports are neither passed nor failed. */
function headline(report: QAReport): string {
  const count = (status: QAReport['status']) =>
    report.checks.filter((check) => check.status === status).length;
  if (report.status === 'skip') return 'QA skipped: this environment cannot be checked';
  if (report.status === 'inconclusive') {
    return `QA inconclusive: ${count('inconclusive')} of ${report.checks.length} checks had no verdict`;
  }
  // A pass counts only passed checks ("4 of 5" with a skipped one); a failure counts the failed ones.
  const against = report.passed ? report.checks.length - count('pass') : count('fail');
  return verdictLine('QA', report.passed, against, report.checks.length, 'check');
}

/** One environment's QA checks: the verdict on top, failed checks first, each with its details inline. */
export function QADrawer({ report, onClose }: { report: QAReport | null; onClose: () => void }) {
  // Keep the last report while the drawer animates closed.
  const [shown, setShown] = useState(report);
  const changed = report?.environment !== shown?.environment || report?.created_at !== shown?.created_at;
  if (report !== null && changed) setShown(report);

  return (
    <Drawer
      isOpen={report !== null}
      onClose={onClose}
      title={shown?.environment ?? ''}
      aria-label="QA report"
    >
      {shown && (
        <div className="flex flex-col gap-4">
          <VerdictBanner
            passed={shown.status === 'pass' ? true : shown.status === 'fail' ? false : null}
            meta={timeAgo(shown.created_at)}
          >
            {headline(shown)}
          </VerdictBanner>
          <TaskRows
            rows={failuresFirst(
              shown.checks.map((check) => ({
                key: check.name,
                label: check.name,
                subline: check.details || undefined,
                status: QA_TASK_STATUS[check.status],
              })),
            )}
          />
        </div>
      )}
    </Drawer>
  );
}
