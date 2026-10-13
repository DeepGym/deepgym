/** Pure helpers for the episode step timeline and the lists that follow the current step. */

import type { Frame } from './types';

const BLOCK_PX = 28;
export const GAP_PX = 2;
/** Zoom levels the timeline steps through with its - and + buttons; 1 is the reset. */
export const ZOOM_LEVELS: readonly number[] = [0.25, 0.5, 1, 2, 4];

export interface TimelineLayout {
  /** Width of one step block in px. */
  block: number;
  /** Distance from one block's start to the next (block + gap). */
  stride: number;
  /** Width of the whole track. */
  width: number;
  /** Whether step numbers fit inside the blocks. */
  labels: boolean;
}

/** Pixel layout of `count` step blocks at `zoom`. */
export function timelineLayout(count: number, zoom: number): TimelineLayout {
  const block = Math.max(4, Math.round(BLOCK_PX * zoom));
  const stride = block + GAP_PX;
  const digits = String(Math.max(0, count - 1)).length;
  return { block, stride, width: Math.max(0, count * stride - GAP_PX), labels: block >= digits * 7 + 6 };
}

/** The next zoom level in `direction`, clamped to the ends of ZOOM_LEVELS. */
export function stepZoom(zoom: number, direction: 1 | -1): number {
  const at = ZOOM_LEVELS.indexOf(zoom);
  return ZOOM_LEVELS[Math.min(ZOOM_LEVELS.length - 1, Math.max(0, at + direction))];
}

export type StepTone = 'good' | 'bad' | 'neutral';

/** Colour of a step: the sign of its reward; the final observation takes the episode's verdict. */
export function stepTone(frame: Frame, passed: boolean | null): StepTone {
  if (frame.reward !== null && frame.reward !== 0) return frame.reward > 0 ? 'good' : 'bad';
  if (frame.action === null && passed !== null) return passed ? 'good' : 'bad';
  return 'neutral';
}

/**
 * Scroll offset that shows the item at [start, start + size) in a viewport `viewport` px long
 * currently scrolled to `scroll`, moving as little as possible.
 */
export function scrollToShow(scroll: number, viewport: number, start: number, size: number): number {
  if (start < scroll) return start;
  if (start + size > scroll + viewport) return start + size - viewport;
  return scroll;
}
