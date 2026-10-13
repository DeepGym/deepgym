import type { ReactNode } from 'react';
import { cx } from '@/utils/cx';

/** Why an action failed, e.g. the API's reason for refusing a run; announced to screen readers. */
export function ActionError({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p
      role="alert"
      className={cx(
        'rounded-2lg bg-background-tertiary-error px-3 py-2 text-body-2-medium text-text-error-primary',
        className,
      )}
    >
      {children}
    </p>
  );
}
