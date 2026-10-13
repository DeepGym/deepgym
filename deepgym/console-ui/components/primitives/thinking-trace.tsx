'use client';
// Adapted from Beautiful UI "Thinking" (https://beautifului.dev), MIT. See ./LICENSE.
// Changes: rows come from a recorded trajectory and are selectable; BoardUI tokens.

import { RiArrowDownSLine, RiCheckLine, RiSparkling2Fill } from '@remixicon/react';
import { useEffect, useRef, useState } from 'react';
import { Collapse } from '@/components/ui/collapse';
import { scrollToShow } from '@/lib/timeline';
import { cx } from '@/utils/cx';

export interface TraceRow {
  key: string;
  primary: string;
  secondary?: string;
}

interface ThinkingTraceProps {
  /** Header text, e.g. "Ran 12 steps". */
  label: string;
  rows: TraceRow[];
  /** Index of the highlighted row. */
  selected: number;
  onSelect: (index: number) => void;
  /** Keep the selected row scrolled into view as it changes. */
  follow: boolean;
}

/** Expandable step trace; with `follow`, the selected row stays in view as it changes. */
export function ThinkingTrace({ label, rows, selected, onSelect, follow }: ThinkingTraceProps) {
  const [open, setOpen] = useState(true);
  const list = useRef<HTMLDivElement>(null);

  // Scroll only the trace list (scrollIntoView would also scroll the page).
  useEffect(() => {
    const box = list.current;
    const row = box?.querySelector<HTMLElement>(`[data-index="${selected}"]`);
    if (!follow || !box || !row) return;
    box.scrollTop = scrollToShow(box.scrollTop, box.clientHeight, row.offsetTop, row.offsetHeight);
  }, [selected, follow]);

  return (
    <div className="flex w-full flex-col">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="-mx-1.5 flex w-fit cursor-pointer items-center gap-2 rounded-lg px-1.5 py-1 outline-none transition-colors hover:bg-background-secondary-hover focus-visible:ring-2 focus-visible:ring-border-focus-ring"
      >
        <RiSparkling2Fill aria-hidden className="size-4 text-accent-400" />
        <span className="text-body-2-medium text-text-secondary">{label}</span>
        <RiArrowDownSLine
          aria-hidden
          className={cx(
            'size-4 text-foreground-icon-secondary transition-transform duration-300',
            open && 'rotate-180',
          )}
        />
      </button>
      <Collapse open={open}>
        <div ref={list} className="scrollbar-thin relative ms-[7px] mt-1 max-h-80 overflow-y-auto ps-4">
          <span aria-hidden className="absolute inset-y-1 start-[3px] w-px bg-separator-border" />
          <div className="flex flex-col gap-0.5 py-1">
            {rows.map((row, index) => {
              const active = index === selected;
              return (
                <button
                  key={row.key}
                  type="button"
                  data-index={index}
                  aria-current={active ? 'step' : undefined}
                  onClick={() => onSelect(index)}
                  className={cx(
                    'flex min-h-7 w-full cursor-pointer items-center gap-2 rounded-md px-1.5 py-0.5 text-start outline-none transition-colors focus-visible:ring-2 focus-visible:ring-border-focus-ring',
                    active ? 'bg-background-tertiary-default' : 'hover:bg-background-secondary-hover',
                  )}
                >
                  {index < selected ? (
                    <RiCheckLine aria-hidden className="size-3.5 shrink-0 text-text-tertiary" />
                  ) : (
                    <span
                      aria-hidden
                      className={cx(
                        'mx-[3px] size-2 shrink-0 rounded-full',
                        active ? 'bg-accent-400' : 'border border-separator-border',
                      )}
                    />
                  )}
                  <span
                    className={cx(
                      'min-w-0 flex-1 truncate font-mono text-caption-1-medium',
                      active ? 'text-text-primary' : 'text-text-secondary',
                    )}
                  >
                    {row.primary}
                  </span>
                  {row.secondary && (
                    <span className="shrink-0 font-mono text-caption-2-medium text-text-tertiary tabular-nums">
                      {row.secondary}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </Collapse>
    </div>
  );
}
