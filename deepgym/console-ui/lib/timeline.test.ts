import { expect, test } from 'bun:test';
import { scrollToShow, stepTone, stepZoom, timelineLayout } from './timeline';
import type { Frame } from './types';

const frame = (reward: number | null, action: Frame['action'] = null): Frame => ({
  index: 0,
  content: '',
  done: false,
  metadata: null,
  action,
  reward,
  has_screenshot: false,
});
const click = {
  action_type: 'click',
  content: '',
  x: 1,
  y: 1,
  to_x: null,
  to_y: null,
  path: null,
  amount: 1,
};

test('layout scales blocks with zoom and drops numbers once they no longer fit', () => {
  expect(timelineLayout(10, 1)).toEqual({ block: 28, stride: 30, width: 298, labels: true });
  expect(timelineLayout(10, 0.25)).toEqual({ block: 7, stride: 9, width: 88, labels: false });
  expect(timelineLayout(1000, 1).labels).toBe(true);
  expect(timelineLayout(1001, 1).labels).toBe(false);
  expect(timelineLayout(0, 1).width).toBe(0);
});

test('zoom steps through the levels and stops at either end', () => {
  expect(stepZoom(1, 1)).toBe(2);
  expect(stepZoom(1, -1)).toBe(0.5);
  expect(stepZoom(4, 1)).toBe(4);
  expect(stepZoom(0.25, -1)).toBe(0.25);
});

test('steps are coloured by exit code, then reward sign; the final frame by the verdict', () => {
  expect(stepTone(frame(0.5, click), null)).toBe('good');
  expect(stepTone(frame(-1, click), true)).toBe('bad');
  expect(stepTone(frame(0, click), true)).toBe('neutral');
  expect(stepTone(frame(null), false)).toBe('bad');
  expect(stepTone(frame(null), null)).toBe('neutral');
  expect(stepTone({ ...frame(0.5, click), metadata: { exit_code: 1 } }, true)).toBe('bad');
  expect(stepTone({ ...frame(0, click), metadata: { exit_code: 0 } }, true)).toBe('neutral');
});

test('scrolls only as far as needed to show the item', () => {
  expect(scrollToShow(100, 200, 150, 30)).toBe(100);
  expect(scrollToShow(100, 200, 40, 30)).toBe(40);
  expect(scrollToShow(100, 200, 290, 30)).toBe(120);
});
