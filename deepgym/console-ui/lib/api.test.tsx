import { afterEach, expect, jest, test } from 'bun:test';
import { act } from 'react';
import { cleanup, fakeFetch, render } from '@/test/dom';
import { useApi } from './api';

afterEach(cleanup);

interface Run {
  id: string;
  status: 'running' | 'completed';
}

let seen: ReturnType<typeof useApi<Run>>;

/** Records what useApi returns for `path`, polling while the run is running. */
function Probe({ path, intervalMs }: { path: string; intervalMs?: number }) {
  seen = useApi<Run>(path, intervalMs, (run) => run.status === 'running');
  return null;
}

const run = (id: string, status: Run['status'] = 'completed') => Response.json({ id, status });

test('a new path starts empty instead of showing the previous path', async () => {
  const held = fakeFetch();
  const { rerender } = await render(<Probe path="/runs/a" />);
  await held[0].reply(run('a'));
  expect(seen.data?.id).toBe('a');

  await rerender(<Probe path="/runs/b" />);
  expect(seen).toMatchObject({ data: undefined, error: undefined, loading: true });
  await held[1].reply(run('b'));
  expect(seen).toMatchObject({ data: { id: 'b' }, loading: false });
});

test("the previous path's late response never shows for the new one", async () => {
  const held = fakeFetch();
  const { rerender } = await render(<Probe path="/runs/a" />);
  await rerender(<Probe path="/runs/b" />);

  await held[0].reply(run('a'));
  expect(seen).toMatchObject({ data: undefined, loading: true });
  await held[1].reply(run('b'));
  expect(seen.data?.id).toBe('b');
});

test("a reload left over from the previous path doesn't replace the new path's data", async () => {
  const held = fakeFetch();
  const { rerender } = await render(<Probe path="/runs/a" />);
  await held[0].reply(run('a'));
  const reloadA = seen.reload;
  await rerender(<Probe path="/runs/b" />);
  await held[1].reply(run('b'));

  await act(async () => void reloadA());
  await held[2].reply(run('a'));
  expect(seen.data?.id).toBe('b');
});

test('an error belongs to the path that failed', async () => {
  const held = fakeFetch();
  const { rerender } = await render(<Probe path="/runs/a" />);
  await held[0].reply(Response.json({ detail: 'Run not found' }, { status: 404 }));
  expect(seen.error).toBe('Run not found');

  await rerender(<Probe path="/runs/b" />);
  expect(seen).toMatchObject({ error: undefined, loading: true });
});

test('an older poll answering after a newer one is ignored, then polling stops', async () => {
  jest.useFakeTimers();
  const held = fakeFetch();
  await render(<Probe path="/runs/a" intervalMs={1000} />);
  await held[0].reply(run('a', 'running'));
  await act(async () => jest.advanceTimersByTime(2000));
  expect(held).toHaveLength(3);

  await held[2].reply(run('a', 'completed'));
  await held[1].reply(run('a', 'running'));
  expect(seen.data?.status).toBe('completed');

  await act(async () => jest.advanceTimersByTime(5000));
  expect(held).toHaveLength(3);
});
