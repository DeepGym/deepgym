import { afterEach, beforeEach, expect, test } from 'bun:test';
import { act } from 'react';
import type { RunDetail } from '@/lib/types';
import { cleanup, fakeFetch, render, settle } from '@/test/dom';
import { navigation } from '@/test/navigation';

// After the navigation mock, so the page gets the recording router.
const { RunDetailPage } = await import('./run-detail-page');

afterEach(cleanup);
beforeEach(() => navigation.reset('/runs/detail/', 'id=r1'));

const run = (id: string): RunDetail => ({
  id,
  group_id: `group-${id}`,
  environment: 'echo',
  mode: 'local',
  provider: 'local',
  reference: false,
  status: 'completed',
  score: 1,
  passed: true,
  error_type: null,
  duration_ms: 10,
  created_at: '2026-10-13T10:00:00Z',
  finished_at: '2026-10-13T10:00:01Z',
  solution: 'print(1)',
  error: null,
  result: null,
});

const REFUSED = 'Custom code runs locally only with --allow-local-exec';

/** The console API for runs r1 and r2; POST /runs answers with `rerun`. */
function serve(rerun: () => Response) {
  const posts: unknown[] = [];
  fakeFetch((url, init) => {
    if (init?.method === 'POST') {
      posts.push(JSON.parse(String(init.body)));
      return rerun();
    }
    const id = url.match(/^\/api\/runs\/(r\d)$/)?.[1];
    return id ? Response.json(run(id)) : undefined;
  });
  return posts;
}

const runAgain = () =>
  [...document.querySelectorAll('button')].find((item) => item.textContent?.trim() === 'Run again');
const alert = () => document.querySelector('[role="alert"]');

async function clickRunAgain() {
  const button = runAgain();
  if (!button) throw new Error('no Run again button');
  await act(async () => button.click());
  await settle();
}

test('a refused rerun shows the API error and stays on the run', async () => {
  const posts = serve(() => Response.json({ detail: REFUSED }, { status: 403 }));
  await render(<RunDetailPage />);
  await settle();
  await clickRunAgain();

  expect(posts).toEqual([{ environments: ['echo'], mode: 'local', solution: 'print(1)' }]);
  expect(alert()?.textContent).toContain(REFUSED);
  expect(navigation.router.push).not.toHaveBeenCalled();
  expect(runAgain()?.disabled).toBe(false);
});

test('a rerun that starts opens the new run, clearing an earlier error', async () => {
  let refuse = true;
  serve(() =>
    refuse
      ? Response.json({ detail: REFUSED }, { status: 403 })
      : Response.json([{ ...run('r2'), status: 'queued' }]),
  );
  await render(<RunDetailPage />);
  await settle();
  await clickRunAgain();
  expect(alert()).not.toBeNull();

  refuse = false;
  await clickRunAgain();
  expect(alert()).toBeNull();
  expect(navigation.router.push.mock.calls).toEqual([['/runs/detail/?id=r2']]);
});

test("a refused rerun's error stays with its run", async () => {
  serve(() => Response.json({ detail: REFUSED }, { status: 403 }));
  const { rerender } = await render(<RunDetailPage />);
  await settle();
  await clickRunAgain();
  expect(alert()).not.toBeNull();

  navigation.reset('/runs/detail/', 'id=r2');
  await rerender(<RunDetailPage />);
  await settle();
  expect(document.body.textContent).toContain('r2');
  expect(alert()).toBeNull();
});
