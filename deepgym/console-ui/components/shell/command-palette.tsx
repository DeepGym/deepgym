'use client';

import {
  type RemixiconComponentType,
  RiAddLine,
  RiFilmLine,
  RiPlayCircleLine,
  RiSearchLine,
  RiStackLine,
} from '@remixicon/react';
import { Command } from 'cmdk';
import { useRouter } from 'next/navigation';
import { Kbd } from '@/components/base/kbd/kbd';
import { useApi } from '@/lib/api';
import { timeAgo } from '@/lib/format';
import { NAV } from '@/lib/nav';
import type { EnvironmentSummary, EpisodePage, RunPage } from '@/lib/types';

interface PaletteItemProps {
  value: string;
  icon: RemixiconComponentType;
  label: string;
  hint?: string;
  onSelect: () => void;
}

const PaletteItem = ({ value, icon: Icon, label, hint, onSelect }: PaletteItemProps) => (
  <Command.Item
    value={value}
    onSelect={onSelect}
    className="flex cursor-pointer items-center gap-2.5 rounded-2lg px-2 py-1.5 text-body-regular text-text-secondary data-[selected=true]:bg-dropdown-item-hover-background data-[selected=true]:text-text-primary"
  >
    <Icon aria-hidden className="size-4 shrink-0 text-foreground-icon-secondary" />
    <span className="min-w-0 flex-1 truncate">{label}</span>
    {hint && <span className="shrink-0 font-mono text-caption-1-medium text-text-tertiary">{hint}</span>}
  </Command.Item>
);

const groupClass =
  '[&_[cmdk-group-heading]]:text-mono-label [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:pt-2.5 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:text-text-tertiary';

/** The ⌘K palette: jump to any page, environment, recent run or episode. */
export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const envs = useApi<EnvironmentSummary[]>(open ? '/environments' : null);
  const runs = useApi<RunPage>(open ? '/runs?limit=20' : null);
  const episodes = useApi<EpisodePage>(open ? '/episodes?limit=20' : null);

  const go = (href: string) => {
    onOpenChange(false);
    router.push(href);
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Command palette"
      overlayClassName="fixed inset-0 z-50 bg-black/50 backdrop-blur-[2px]"
      contentClassName="fixed top-[12vh] left-1/2 z-50 w-[min(640px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-2xl border border-border-button-default bg-background-primary-default shadow-xl"
    >
      <div className="flex items-center gap-2 border-b border-separator-border px-3.5">
        <RiSearchLine aria-hidden className="size-4.5 text-foreground-icon-secondary" />
        <Command.Input
          placeholder="Jump to a page, environment, run or episode"
          className="h-12 w-full bg-transparent text-body-regular text-text-primary outline-none placeholder:text-text-tertiary"
        />
        <Kbd>esc</Kbd>
      </div>
      <Command.List className="scrollbar-thin max-h-[60vh] overflow-y-auto p-1.5">
        <Command.Empty className="px-3 py-8 text-center text-body-regular text-text-tertiary">
          No matches
        </Command.Empty>
        <Command.Group heading="Pages" className={groupClass}>
          {NAV.map((item) => (
            <PaletteItem
              key={item.href}
              value={`page ${item.label}`}
              icon={item.icon}
              label={item.label}
              hint={item.chord}
              onSelect={() => go(item.href)}
            />
          ))}
          <PaletteItem
            value="page new run"
            icon={RiAddLine}
            label="New run"
            hint="n"
            onSelect={() => go('/runs/?new=1')}
          />
        </Command.Group>
        <Command.Group heading="Environments" className={groupClass}>
          {envs.data?.map((env) => (
            <PaletteItem
              key={env.id}
              value={`env ${env.id} ${env.name}`}
              icon={RiStackLine}
              label={env.name}
              hint={env.id}
              onSelect={() => go(`/environments/?env=${encodeURIComponent(env.id)}`)}
            />
          ))}
        </Command.Group>
        <Command.Group heading="Recent runs" className={groupClass}>
          {runs.data?.runs.map((run) => (
            <PaletteItem
              key={run.id}
              value={`run ${run.id} ${run.environment} ${run.status}`}
              icon={RiPlayCircleLine}
              label={`${run.environment} - ${run.status}`}
              hint={timeAgo(run.created_at)}
              onSelect={() => go(`/runs/detail/?id=${run.id}`)}
            />
          ))}
        </Command.Group>
        <Command.Group heading="Episodes" className={groupClass}>
          {episodes.data?.episodes.map((episode) => (
            <PaletteItem
              key={episode.id}
              value={`episode ${episode.id} ${episode.environment}`}
              icon={RiFilmLine}
              label={`${episode.environment || 'episode'} - ${episode.steps} steps`}
              hint={timeAgo(episode.created_at)}
              onSelect={() => go(`/episodes/?id=${episode.id}`)}
            />
          ))}
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
