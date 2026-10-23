'use client';

import { useState } from 'react';
import { useApi } from './api';

const PAGE_SIZE = 25;

/**
 * Fetch `path` a server page at a time, refetching every `intervalMs`, with the paging a DataTable
 * takes as `serverPaging`. The page is kept with the `query` it pages through, so a new query
 * starts at the first page.
 */
export function usePagedApi<T extends { total: number }>(path: string, intervalMs: number, query = '') {
  const [page, setPage] = useState({ query, index: 0 });
  const pageIndex = page.query === query ? page.index : 0;
  const params = new URLSearchParams(query);
  params.set('offset', String(pageIndex * PAGE_SIZE));
  params.set('limit', String(PAGE_SIZE));
  const { data, error } = useApi<T>(`${path}?${params}`, intervalMs);
  const pageCount = Math.ceil((data?.total ?? 0) / PAGE_SIZE);
  // Rows that finish or go away between polls can leave this page past the last one, empty and
  // with no pager to leave it: move to the last page there is. Set while rendering, so React
  // re-renders at once instead of showing the empty page first.
  const lastPage = Math.max(pageCount - 1, 0);
  if (data && pageIndex > lastPage) setPage({ query, index: lastPage });
  return {
    data,
    error,
    paging: { pageIndex, pageCount, onPageChange: (index: number) => setPage({ query, index }) },
  };
}
