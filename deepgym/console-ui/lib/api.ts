'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

/** FastAPI's `detail` message for a failed response. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body: { detail?: unknown } = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg ?? String(item)).join('; ');
  } catch {
    // Not JSON: fall back to the status text below.
  }
  return response.statusText || `HTTP ${response.status}`;
}

/**
 * The per-launch token the console writes into each page; every API call must send it.
 * `bun run dev` pages come from Next instead, so they read NEXT_PUBLIC_DEEPGYM_TOKEN.
 */
const token = (): string =>
  document.querySelector<HTMLMetaElement>('meta[name="deepgym-token"]')?.content ??
  process.env.NEXT_PUBLIC_DEEPGYM_TOKEN ??
  '';

/** Fetch `/api{path}` with the console token; throws an Error with the API's message on a non-2xx response. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set('X-DeepGym-Token', token());
  if (init.body) headers.set('Content-Type', 'application/json');
  const response = await fetch(`/api${path}`, { ...init, headers });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

/** Call `/api{path}`: JSON in, JSON out. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await apiFetch(path, init);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

/** Message of anything thrown, for showing next to the action that failed. */
export const errorText = (caught: unknown): string =>
  caught instanceof Error ? caught.message : String(caught);

/** URL path for an environment id, keeping its slashes (e.g. "cyber/web_path_traversal_toy"). */
export const envPath = (id: string): string => id.split('/').map(encodeURIComponent).join('/');

interface ApiState<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
  reload: () => Promise<void>;
}

/** The newest response applied, and the path it answers. */
interface Answer<T> {
  path: string;
  data?: T;
  error?: string;
}

/**
 * Fetch `path` (skipped while null) and refetch every `intervalMs` if given,
 * for as long as `keepPolling(data)` holds. Data and errors belong to the path
 * they answer, so a new path starts empty. Responses for a path the component
 * has since moved off, or older than one already applied (an overtaken poll),
 * are dropped.
 */
export function useApi<T>(
  path: string | null,
  intervalMs?: number,
  keepPolling?: (data: T) => boolean,
): ApiState<T> {
  const [answer, setAnswer] = useState<Answer<T>>();
  const own = answer?.path === path ? answer : undefined;
  const current = useRef(path);
  current.current = path;
  // Requests are numbered in the order sent; `applied` is the newest one shown.
  const sent = useRef(0);
  const applied = useRef(0);
  const latest = useRef({ data: own?.data, keepPolling });
  latest.current = { data: own?.data, keepPolling };

  const reload = useCallback(async () => {
    if (path === null) return;
    const request = ++sent.current;
    let update: (previous: Answer<T> | undefined) => Answer<T>;
    try {
      const data = await api<T>(path);
      update = () => ({ path, data });
    } catch (caught) {
      const error = errorText(caught);
      // A failed refresh keeps the path's last data next to the error.
      update = (previous) => ({ path, data: previous?.path === path ? previous.data : undefined, error });
    }
    if (current.current !== path || request < applied.current) return;
    applied.current = request;
    setAnswer(update);
  }, [path]);

  useEffect(() => {
    if (path === null) return;
    void reload();
    if (!intervalMs) return;
    const timer = setInterval(() => {
      const { data: last, keepPolling: shouldPoll } = latest.current;
      if (last === undefined || !shouldPoll || shouldPoll(last)) void reload();
    }, intervalMs);
    return () => clearInterval(timer);
  }, [path, intervalMs, reload]);

  return { data: own?.data, error: own?.error, loading: path !== null && own === undefined, reload };
}
