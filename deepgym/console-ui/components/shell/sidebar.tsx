'use client';

import { RiBookOpenLine } from '@remixicon/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Kbd } from '@/components/base/kbd/kbd';
import { isActive, NAV } from '@/lib/nav';
import { cx } from '@/utils/cx';
import { BrandMark } from './brand-mark';

/** Floating left navigation panel (md and up), in BoardUI's sidebar recipe. */
export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside
      aria-label="Console navigation"
      className="flex h-full w-full flex-col rounded-3xl border border-border-button-default bg-background-secondary-default p-3 shadow-sidebar"
    >
      <div className="flex items-center gap-2.5 px-1.5 pt-1 pb-5">
        <span className="flex size-8 items-center justify-center rounded-2lg bg-linear-to-b from-accent-500 to-accent-600 shadow-nav-selected">
          <BrandMark className="size-4.5 text-text-white" />
        </span>
        <div className="leading-tight">
          <p className="font-display text-body-medium text-text-primary">DeepGym</p>
          <p className="text-mono-label text-text-tertiary">Console</p>
        </div>
      </div>
      <p className="text-mono-label px-2 pb-1.5 text-text-tertiary">Workspace</p>
      <nav className="flex flex-col gap-0.5">
        {NAV.map((item) => {
          const active = isActive(item.href, pathname);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? 'page' : undefined}
              className={cx(
                'group flex items-center gap-2 rounded-2lg p-2 outline-none transition-colors focus-visible:ring-2 focus-visible:ring-border-focus-ring',
                active
                  ? 'bg-linear-to-b from-accent-500 to-accent-600 shadow-nav-selected'
                  : 'hover:bg-background-secondary-hover',
              )}
            >
              <item.icon
                aria-hidden
                className={cx(
                  'size-5 shrink-0',
                  active ? 'text-text-white' : 'text-foreground-icon-secondary',
                )}
              />
              <span
                className={cx('flex-1 text-body-medium', active ? 'text-text-white' : 'text-text-secondary')}
              >
                {item.label}
              </span>
              {!active && <Kbd className="hidden group-hover:inline-flex">{item.chord}</Kbd>}
            </Link>
          );
        })}
      </nav>
      <a
        href="/docs"
        className="mt-auto flex items-center gap-2 rounded-2lg p-2 text-body-2-medium text-text-secondary outline-none hover:bg-background-secondary-hover focus-visible:ring-2 focus-visible:ring-border-focus-ring"
      >
        <RiBookOpenLine aria-hidden className="size-4 text-foreground-icon-secondary" />
        API reference
      </a>
    </aside>
  );
}
