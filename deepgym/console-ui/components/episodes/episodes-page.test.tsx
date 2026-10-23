import { afterEach, beforeEach, expect, test } from 'bun:test';
import { act } from 'react';
import type { EpisodeDetail, EpisodePage, EpisodeSummary } from '@/lib/types';
import { cleanup, fakeFetch, render, settle } from '@/test/dom';
import { navigation } from '@/test/navigation';

// After the navigation mock, so the page reads ?id= from it.
const { EpisodesPage } = await import('./episodes-page');

afterEach(cleanup);
beforeEach(() => navigation.reset('/episodes/'));

const episode = (id: string, extra: Partial<EpisodeSummary> = {}): EpisodeSummary => ({
  id,
  environment: `env-${id}`,
  steps: 2,
  total_reward: 0,
  score: 1,
  passed: true,
  has_verdict: true,
  error_type: null,
  created_at: '2026-10-23T10:00:00Z',
  ...extra,
});

/** The provider failed before the verifier judged it: no score, no pass. */
const crashed = episode('crashed', {
  score: null,
  passed: null,
  has_verdict: false,
  error_type: 'provider_error',
});
/** Judged and failed: a real 0. */
const zero = episode('zero', { score: 0, passed: false });

const page = (episodes: EpisodeSummary[], total: number) =>
  Response.json({ episodes, total } satisfies EpisodePage);

const rowText = (text: string) =>
  [...document.querySelectorAll('[role="row"]')].find((row) => row.textContent?.includes(text))?.textContent;

test('an episode without a verdict is listed as no verdict with its error, not as a failed 0', async () => {
  fakeFetch(() => page([crashed, zero], 2));
  await render(<EpisodesPage />);
  await settle();

  expect(rowText('env-crashed')).toContain('no verdictprovider_error');
  expect(rowText('env-crashed')).not.toContain('fail');
  expect(rowText('env-zero')).toContain('0.00fail');
});

test('replaying an episode without a verdict shows its error and result output, not a failed 0', async () => {
  navigation.reset('/episodes/', 'id=crashed');
  const detail: EpisodeDetail = {
    ...crashed,
    frames: [
      {
        index: 0,
        content: '',
        done: true,
        metadata: null,
        action: null,
        reward: null,
        has_screenshot: false,
      },
    ],
    result: {
      score: 0,
      passed: false,
      output: 'provider unreachable',
      stderr: '',
      exit_code: 1,
      execution_time_ms: 0,
      sandbox_id: 'error',
      reward_components: null,
      error_type: 'provider_error',
      cases: null,
    },
  };
  fakeFetch((url) => (url === '/api/episodes/crashed' ? Response.json(detail) : page([crashed], 1)));
  await render(<EpisodesPage />);
  await settle();

  const card = [...document.querySelectorAll('section')].find((section) =>
    section.textContent?.startsWith('Final score'),
  );
  expect(card?.textContent).toBe('Final score-no verdictprovider_errorprovider unreachable');
});

test('every episode is reachable: the pager fetches the next page from the server', async () => {
  const newest = Array.from({ length: 25 }, (_, index) => episode(`new${index}`));
  const requested: string[] = [];
  fakeFetch((url) => {
    requested.push(url);
    return url.includes('offset=25') ? page([episode('oldest')], 26) : page(newest, 26);
  });
  await render(<EpisodesPage />);
  await settle();
  expect(rowText('env-oldest')).toBeUndefined();

  const next = document.querySelector<HTMLButtonElement>('button[aria-label="Go to page 2"]');
  if (!next) throw new Error('no pager');
  await act(async () => next.click());
  await settle();

  expect(requested).toEqual(['/api/episodes?offset=0&limit=25', '/api/episodes?offset=25&limit=25']);
  expect(rowText('env-oldest')).toContain('env-oldest');
  expect(rowText('env-new0')).toBeUndefined();
});
