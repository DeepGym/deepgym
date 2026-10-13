import type { ReactNode } from 'react';

/** Animated show/hide (height and fade). Hidden content is inert, so it can't take focus. */
export function Collapse({ open, id, children }: { open: boolean; id?: string; children: ReactNode }) {
  return (
    <div
      id={id}
      inert={!open}
      className="grid transition-[grid-template-rows,opacity] duration-300"
      style={{ gridTemplateRows: open ? '1fr' : '0fr', opacity: open ? 1 : 0 }}
    >
      <div className="overflow-hidden">{children}</div>
    </div>
  );
}
