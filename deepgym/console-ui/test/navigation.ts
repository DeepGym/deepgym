import { mock } from 'bun:test';

/**
 * What next/navigation hands components in tests: a router that records navigations, and the
 * current path and query. Import this before the component under test, and `reset` per test.
 */
export const navigation = {
  router: {
    push: mock((_href: string) => {}),
    replace: mock((_href: string, _options?: { scroll: boolean }) => {}),
  },
  pathname: '/',
  search: new URLSearchParams(),
  reset(pathname: string, search = '') {
    navigation.pathname = pathname;
    navigation.search = new URLSearchParams(search);
    navigation.router.push.mockClear();
    navigation.router.replace.mockClear();
  },
};

mock.module('next/navigation', () => ({
  useRouter: () => navigation.router,
  usePathname: () => navigation.pathname,
  useSearchParams: () => navigation.search,
}));
