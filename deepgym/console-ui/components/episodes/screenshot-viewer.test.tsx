import { afterEach, beforeEach, expect, spyOn, test } from 'bun:test';
import { act } from 'react';
import type { Frame } from '@/lib/types';
import { cleanup, fakeFetch, render, settle } from '@/test/dom';
import { ScreenshotViewer } from './screenshot-viewer';

afterEach(cleanup);

let revoked: string[];
beforeEach(() => {
  let created = 0;
  revoked = [];
  spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:shot-${++created}`);
  spyOn(URL, 'revokeObjectURL').mockImplementation((url: string) => {
    revoked.push(url);
  });
});

const frame = (index: number): Frame => ({
  index,
  content: '',
  done: false,
  metadata: null,
  reward: null,
  has_screenshot: true,
  action: { action_type: 'click', content: '', x: 50, y: 25, to_x: null, to_y: null, path: null, amount: 0 },
});

const png = () => new Response(new Blob(['png'], { type: 'image/png' }));
const image = (step: number) => document.querySelector<HTMLImageElement>(`img[alt="Screen at step ${step}"]`);
/** The circle drawn where the frame's action points. */
const marker = () => document.querySelector('span[aria-hidden]');

/** Fire the step's image load, as the browser does once it has decoded the screenshot. */
async function decode(step: number) {
  const img = image(step);
  if (!img) throw new Error(`no image for step ${step}`);
  Object.defineProperties(img, { naturalWidth: { value: 100 }, naturalHeight: { value: 50 } });
  await act(async () => {
    img.dispatchEvent(new Event('load'));
  });
}

test("a new frame waits for its own screenshot, never showing the previous one's", async () => {
  const held = fakeFetch();
  const { rerender } = await render(<ScreenshotViewer episodeId="ep" frame={frame(0)} />);
  expect(held[0].url).toBe('/api/episodes/ep/frames/0/screenshot');
  await held[0].reply(png());
  expect(image(0)?.getAttribute('src')).toBe('blob:shot-1');
  expect(marker()).toBeNull();
  await decode(0);
  expect(marker()).not.toBeNull();

  await rerender(<ScreenshotViewer episodeId="ep" frame={frame(1)} />);
  expect(revoked).toEqual(['blob:shot-1']);
  expect(image(1)).toBeNull();
  expect(document.body.textContent).toContain('Loading screenshot');
  expect(marker()).toBeNull();

  await held[1].reply(png());
  expect(image(1)?.getAttribute('src')).toBe('blob:shot-2');
  expect(marker()).toBeNull();
  await decode(1);
  expect(marker()).not.toBeNull();
});

test('the same step of another episode loads its own screenshot', async () => {
  const held = fakeFetch();
  const { rerender } = await render(<ScreenshotViewer episodeId="ep" frame={frame(0)} />);
  await held[0].reply(png());
  await decode(0);

  await rerender(<ScreenshotViewer episodeId="other" frame={frame(0)} />);
  await settle();
  expect(held[1].url).toBe('/api/episodes/other/frames/0/screenshot');
  expect(document.querySelector('img[src="blob:shot-1"]')).toBeNull();
  expect(document.body.textContent).toContain('Loading screenshot');
  expect(marker()).toBeNull();
});

test('a screenshot that fails to load shows why, with no image or marker', async () => {
  fakeFetch(() => Response.json({ detail: 'Screenshot file is missing' }, { status: 404 }));
  await render(<ScreenshotViewer episodeId="ep" frame={frame(0)} />);
  await settle();
  expect(document.body.textContent).toContain('Screenshot file is missing');
  expect(document.querySelector('img')).toBeNull();
  expect(marker()).toBeNull();
});
