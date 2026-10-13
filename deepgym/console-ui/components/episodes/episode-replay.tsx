'use client';

import { RiArrowLeftSLine, RiArrowRightSLine, RiPauseLine, RiPlayLine } from '@remixicon/react';
import { useEffect, useState } from 'react';
import { Checkbox } from 'react-aria-components';
import { Chip } from '@/components/base/badges/chip';
import { Button } from '@/components/base/buttons/button';
import { CheckboxGlyph } from '@/components/base/checkbox/checkbox-glyph';
import { Kbd } from '@/components/base/kbd/kbd';
import { CodeBlock } from '@/components/primitives/code-block';
import { ThinkingTrace } from '@/components/primitives/thinking-trace';
import { Card } from '@/components/ui/card';
import { PassBadge } from '@/components/ui/status-badge';
import { formatScore } from '@/lib/format';
import type { EpisodeDetail } from '@/lib/types';
import { useHotkey } from '@/lib/use-hotkey';
import { describeAction } from './describe-action';
import { ScreenshotViewer } from './screenshot-viewer';
import { StepTimeline } from './step-timeline';

const PLAY_INTERVAL_MS = 900;

/** Step-by-step replay: a timeline of steps, then the screenshot, observation text, action taken and its reward. */
export function EpisodeReplay({ episode }: { episode: EpisodeDetail }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [follow, setFollow] = useState(true);
  const last = episode.frames.length - 1;
  const frame = episode.frames[index];
  const step = (delta: number) => setIndex((current) => Math.min(last, Math.max(0, current + delta)));

  useHotkey('ArrowRight', () => step(1));
  useHotkey('ArrowLeft', () => step(-1));
  useHotkey('l', () => step(1));
  useHotkey('h', () => step(-1));
  useHotkey(' ', () => setPlaying((value) => !value));

  useEffect(() => {
    if (!playing) return;
    if (index >= last) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setIndex((current) => current + 1), PLAY_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [playing, index, last]);

  if (!frame) return <p className="text-body-regular text-text-tertiary">This episode has no frames.</p>;

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="small"
          variant="secondary"
          iconOnly
          leadingIcon={RiArrowLeftSLine}
          aria-label="Previous step"
          disabled={index === 0}
          onClick={() => step(-1)}
        />
        <Button
          size="small"
          variant="secondary"
          iconOnly
          leadingIcon={playing ? RiPauseLine : RiPlayLine}
          aria-label={playing ? 'Pause' : 'Play'}
          onClick={() => setPlaying((value) => !value)}
        />
        <Button
          size="small"
          variant="secondary"
          iconOnly
          leadingIcon={RiArrowRightSLine}
          aria-label="Next step"
          disabled={index === last}
          onClick={() => step(1)}
        />
        <StepTimeline
          frames={episode.frames}
          passed={episode.passed}
          index={index}
          onSelect={setIndex}
          follow={follow}
        />
        <span className="font-mono text-caption-1-medium text-text-secondary tabular-nums">
          {index}/{last}
        </span>
      </div>
      <div className="-mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
        <Checkbox
          isSelected={follow}
          onChange={setFollow}
          className="flex cursor-pointer items-center gap-1.5 text-caption-1-medium text-text-secondary outline-none"
        >
          {(state) => (
            <>
              <CheckboxGlyph size="sm" state={state} />
              Auto-scroll
            </>
          )}
        </Checkbox>
        <span className="hidden items-center gap-1 text-caption-1-medium text-text-tertiary md:flex">
          <Kbd>←</Kbd>
          <Kbd>→</Kbd> step <Kbd>space</Kbd> play
        </span>
      </div>
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_20rem]">
        <ScreenshotViewer episodeId={episode.id} frame={frame} />
        <div className="flex min-w-0 flex-col gap-3">
          <Card title="Action">
            <p className="font-mono text-body-2-medium break-words text-text-primary">
              {describeAction(frame.action)}
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {frame.reward !== null && (
                <Chip variant="caption" color="blue">
                  reward {formatScore(frame.reward)}
                </Chip>
              )}
              {frame.done && (
                <Chip variant="caption" color="neutral">
                  done
                </Chip>
              )}
            </div>
          </Card>
          <ThinkingTrace
            label={`Ran ${episode.steps} steps`}
            rows={episode.frames.map((item) => ({
              key: String(item.index),
              primary: describeAction(item.action),
              secondary: item.reward === null ? undefined : formatScore(item.reward),
            }))}
            selected={index}
            onSelect={setIndex}
            follow={follow}
          />
          {episode.result && (
            <Card title="Final score">
              <div className="flex items-center gap-3">
                <span className="font-mono text-title-2-semibold text-text-primary">
                  {formatScore(episode.result.score)}
                </span>
                <PassBadge passed={episode.result.passed} />
              </div>
              {episode.result.output && (
                <p className="mt-1.5 text-body-2-regular text-text-secondary">{episode.result.output}</p>
              )}
            </Card>
          )}
        </div>
      </div>
      <CodeBlock code={frame.content || '(no output)'} filename={`observation ${index}`} wrap plain />
    </div>
  );
}
