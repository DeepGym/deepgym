'use client';

import { RiZoomInLine, RiZoomOutLine } from '@remixicon/react';
import { type KeyboardEvent, type MouseEvent, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/base/buttons/button';
import { formatScore } from '@/lib/format';
import {
  GAP_PX,
  type StepTone,
  scrollToShow,
  stepTone,
  stepZoom,
  timelineLayout,
  ZOOM_LEVELS,
} from '@/lib/timeline';
import type { Frame } from '@/lib/types';
import { cx } from '@/utils/cx';
import { describeAction } from './describe-action';

const TONE: Record<StepTone, string> = {
  good: 'bg-status-lime-background text-status-lime-text',
  bad: 'bg-status-rose-background text-status-rose-text',
  neutral: 'bg-background-tertiary-default text-text-tertiary',
};

interface StepTimelineProps {
  frames: Frame[];
  /** Episode verdict, which colours the final observation. */
  passed: boolean | null;
  index: number;
  onSelect: (index: number) => void;
  /** Keep the current step scrolled into view. */
  follow: boolean;
}

/**
 * Scrubber of numbered step blocks coloured by reward, with a playhead on the current step.
 * Click a block to jump; Home/End/PageUp/PageDown move when it has focus (arrows step globally).
 */
export function StepTimeline({ frames, passed, index, onSelect, follow }: StepTimelineProps) {
  const [zoom, setZoom] = useState(1);
  const track = useRef<HTMLDivElement>(null);
  const layout = timelineLayout(frames.length, zoom);
  const last = frames.length - 1;

  useEffect(() => {
    const box = track.current;
    if (!follow || !box) return;
    box.scrollLeft = scrollToShow(box.scrollLeft, box.clientWidth, index * layout.stride, layout.block);
  }, [index, follow, layout.stride, layout.block]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const keys: Record<string, number> = { Home: 0, End: last, PageUp: index - 10, PageDown: index + 10 };
    if (!(event.key in keys)) return;
    event.preventDefault();
    onSelect(Math.min(last, Math.max(0, keys[event.key])));
  };

  const onClick = (event: MouseEvent<HTMLDivElement>) => {
    const block =
      event.target instanceof HTMLElement ? event.target.closest<HTMLElement>('[data-step]') : null;
    if (block) onSelect(Number(block.dataset.step));
  };

  return (
    <div className="flex min-w-48 flex-1 items-center gap-1.5">
      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Step"
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={index}
        aria-valuetext={`Step ${index} of ${last}: ${describeAction(frames[index]?.action ?? null)}`}
        onKeyDown={onKeyDown}
        onClick={onClick}
        className="scrollbar-thin min-w-0 flex-1 cursor-pointer overflow-x-auto rounded-lg py-1 outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring"
      >
        <div className="relative flex h-7" style={{ width: layout.width, gap: GAP_PX }}>
          {frames.map((frame, step) => (
            <span
              key={frame.index}
              data-step={step}
              title={`Step ${step}: ${describeAction(frame.action)}${frame.reward === null ? '' : `, reward ${formatScore(frame.reward)}`}`}
              className={cx(
                'flex h-full shrink-0 items-center justify-center rounded-sm font-mono text-caption-2-medium tabular-nums transition-colors hover:brightness-125',
                TONE[stepTone(frame, passed)],
                step === index && 'text-text-primary',
              )}
              style={{ width: layout.block }}
            >
              {layout.labels && step}
            </span>
          ))}
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-y-1 left-0 w-0.5 rounded-full bg-accent-400 shadow-[0_0_0_2px_color-mix(in_srgb,var(--color-accent-400)_25%,transparent)] transition-transform duration-150"
            style={{ transform: `translateX(${index * layout.stride + layout.block / 2 - 1}px)` }}
          />
        </div>
      </div>
      <Button
        size="small"
        variant="secondary"
        iconOnly
        leadingIcon={RiZoomOutLine}
        aria-label="Zoom out"
        disabled={zoom === ZOOM_LEVELS[0]}
        onClick={() => setZoom(stepZoom(zoom, -1))}
      />
      <Button
        size="small"
        variant="secondary"
        aria-label={`Zoom ${zoom}x, reset to 1x`}
        onClick={() => setZoom(1)}
        className="min-w-11 font-mono tabular-nums"
      >
        {zoom}x
      </Button>
      <Button
        size="small"
        variant="secondary"
        iconOnly
        leadingIcon={RiZoomInLine}
        aria-label="Zoom in"
        disabled={zoom === ZOOM_LEVELS.at(-1)}
        onClick={() => setZoom(stepZoom(zoom, 1))}
      />
    </div>
  );
}
