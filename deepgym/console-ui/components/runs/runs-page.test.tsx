import { afterEach, beforeEach, expect, test } from 'bun:test';
import { act } from 'react';
import type { EnvironmentSummary, RunPage, RunSummary } from '@/lib/types';
import { cleanup, fakeFetch, render, settle } from '@/test/dom';
import { navigation } from '@/test/navigation';

// After the navigation mock, so the page gets the recording router.
const { RunsPage } = await import('./runs-page');

afterEach(cleanup);
beforeEach(() => navigation.reset('/runs/', 'new=echo'));

const env = (id: string): EnvironmentSummary => ({
  id,
  name: id,
  kind: 'single-turn',
  type: 'code',
  difficulty: 'easy',
  domain: 'coding',
  family: '',
  tags: [],
  variants: 1,
});

const created = (id: string, environment: string): RunSummary => ({
  id,
  group_id: 'g1',
  environment,
  mode: 'local',
  provider: 'local',
  reference: true,
  status: 'queued',
  score: null,
  passed: null,
  error_type: null,
  duration_ms: null,
  created_at: '2026-10-13T10:00:00Z',
  finished_at: null,
});

const page = (runs: RunSummary[], total: number) => Response.json({ runs, total } satisfies RunPage);

/** The console API: POST /runs starts one run per environment; none are listed; stats never arrive. */
function serve() {
  const posts: string[][] = [];
  fakeFetch((url, init) => {
    if (init?.method === 'POST') {
      const { environments }: { environments: string[] } = JSON.parse(String(init.body));
      posts.push(environments);
      return Response.json(environments.map((id, index) => created(`r${index + 1}`, id)));
    }
    if (url.startsWith('/api/environments')) return Response.json([env('echo'), env('reverse')]);
    if (url === '/api/runs/config') return Response.json({ local_custom_code: false });
    if (url.startsWith('/api/runs/stats')) return undefined;
    return page([], 0);
  });
  return posts;
}

const button = (text: string) =>
  [...document.querySelectorAll('button')].find((item) => item.textContent?.trim() === text);

async function click(target: Element | null | undefined) {
  if (!(target instanceof HTMLElement)) throw new Error('nothing to click');
  await act(async () => target.click());
  await settle();
}

test('starting one run opens it, and nothing navigates after', async () => {
  const posts = serve();
  await render(<RunsPage />);
  await settle();
  await click(button('Start 1 run'));

  expect(posts).toEqual([['echo']]);
  expect(navigation.router.push.mock.calls).toEqual([['/runs/detail/?id=r1']]);
  expect(navigation.router.replace).not.toHaveBeenCalled();
  // Still submitting until that navigation unmounts the dialog, so it can't start the run twice.
  expect(button('Starting')?.disabled).toBe(true);
});

test('starting an eval shows its group, and nothing navigates after', async () => {
  const posts = serve();
  await render(<RunsPage />);
  await settle();
  await click(document.querySelector('[role="option"][data-key="reverse"]'));
  await click(button('Start 2 runs'));

  expect(posts).toEqual([['echo', 'reverse']]);
  expect(navigation.router.replace.mock.calls).toEqual([['/runs/?group=g1']]);
  expect(navigation.router.push).not.toHaveBeenCalled();
});

test('cancel closes the dialog without starting anything', async () => {
  const posts = serve();
  await render(<RunsPage />);
  await settle();
  await click(button('Cancel'));

  expect(posts).toEqual([]);
  expect(navigation.router.replace.mock.calls).toEqual([['/runs/', { scroll: false }]]);
  expect(navigation.router.push).not.toHaveBeenCalled();
});

const rowText = (text: string) =>
  [...document.querySelectorAll('[role="row"]')].find((row) => row.textContent?.includes(text))?.textContent;

test('every run is reachable: the pager fetches the next page, and a new filter starts at the first', async () => {
  navigation.reset('/runs/');
  const newest = Array.from({ length: 25 }, (_, index) => created(`r${index}`, `new${index}`));
  const requested: string[] = [];
  fakeFetch((url) => {
    if (!url.startsWith('/api/runs?')) return undefined;
    requested.push(url);
    return url.includes('offset=25') ? page([created('r25', 'oldest')], 26) : page(newest, 26);
  });
  const { rerender } = await render(<RunsPage />);
  await settle();
  expect(rowText('oldest')).toBeUndefined();

  await click(document.querySelector('button[aria-label="Go to page 2"]'));
  expect(rowText('oldest')).toContain('oldest');
  expect(rowText('new0')).toBeUndefined();

  navigation.search = new URLSearchParams('status=failed');
  await rerender(<RunsPage />);
  await settle();

  expect(requested).toEqual([
    '/api/runs?&offset=0&limit=25',
    '/api/runs?&offset=25&limit=25',
    '/api/runs?status=failed&offset=0&limit=25',
  ]);
});
