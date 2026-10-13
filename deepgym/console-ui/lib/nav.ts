import {
  type RemixiconComponentType,
  RiBox3Line,
  RiDashboardLine,
  RiFilmLine,
  RiPlayCircleLine,
  RiShieldCheckLine,
  RiStackLine,
} from '@remixicon/react';

export interface NavItem {
  href: string;
  label: string;
  icon: RemixiconComponentType;
  /** Two-key shortcut, e.g. "g r". */
  chord: string;
}

/** The console's pages, shared by the sidebar, the command palette and the shortcuts. */
export const NAV: NavItem[] = [
  { href: '/', label: 'Overview', icon: RiDashboardLine, chord: 'g o' },
  { href: '/environments/', label: 'Environments', icon: RiStackLine, chord: 'g e' },
  { href: '/runs/', label: 'Runs', icon: RiPlayCircleLine, chord: 'g r' },
  { href: '/episodes/', label: 'Episodes', icon: RiFilmLine, chord: 'g p' },
  { href: '/sandboxes/', label: 'Sandboxes', icon: RiBox3Line, chord: 'g s' },
  { href: '/qa/', label: 'QA', icon: RiShieldCheckLine, chord: 'g q' },
];

/** True if `pathname` is `href` or one of its sub-pages. */
export function isActive(href: string, pathname: string): boolean {
  return href === '/' ? pathname === '/' : pathname.startsWith(href.replace(/\/$/, ''));
}
