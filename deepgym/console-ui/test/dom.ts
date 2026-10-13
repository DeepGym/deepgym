import { jest, mock, spyOn } from 'bun:test';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const roots = new Set<Root>();

/** Render `ui` into a fresh container and run its effects. */
export async function render(ui: ReactNode) {
  const container = document.body.appendChild(document.createElement('div'));
  const root = createRoot(container);
  roots.add(root);
  await act(async () => root.render(ui));
  return { container, rerender: (next: ReactNode) => act(async () => root.render(next)) };
}

/** Unmount everything `render` mounted (stopping its polling), empty the page, undo spies and fake timers. */
export function cleanup() {
  act(() => {
    for (const root of roots) root.unmount();
  });
  roots.clear();
  document.body.replaceChildren();
  mock.restore();
  jest.useRealTimers();
}

/**
 * Let answered fetches, their body parsing and the renders they cause finish. setImmediate runs
 * once all pending promise callbacks have, and bun's fake timers leave it real.
 */
export const settle = () => act(() => new Promise<void>((resolve) => setImmediate(resolve)));

/** A request the code under test is still waiting on. */
export interface HeldFetch {
  url: string;
  init: RequestInit | undefined;
  /** Answer it, then settle. */
  reply: (response: Response) => Promise<void>;
}

/**
 * Stand in for fetch. `answer` replies at once; requests it returns undefined for are held,
 * in the order they were made, for the test to answer in any order.
 */
export function fakeFetch(
  answer: (url: string, init?: RequestInit) => Response | undefined = () => undefined,
): HeldFetch[] {
  const held: HeldFetch[] = [];
  const fake = (input: string | URL | Request, init?: RequestInit) =>
    new Promise<Response>((resolve) => {
      const url = String(input);
      const response = answer(url, init);
      if (response) return resolve(response);
      held.push({
        url,
        init,
        reply: async (reply) => {
          resolve(reply);
          await settle();
        },
      });
    });
  // Bun's fetch type adds fetch.preconnect, which the console never calls.
  spyOn(globalThis, 'fetch').mockImplementation(fake as typeof fetch);
  return held;
}
