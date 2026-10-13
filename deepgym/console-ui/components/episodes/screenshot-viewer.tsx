'use client';

import { RiImageLine } from '@remixicon/react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { apiFetch, errorText } from '@/lib/api';
import type { Frame } from '@/lib/types';
import { cx } from '@/utils/cx';
import { actionPoint } from './describe-action';

interface Screenshot {
  /** The API path it came from, which names its episode and frame. */
  path: string;
  url?: string;
  error?: string;
}

/**
 * Object URL of the screenshot at `path`, fetched with the console token (an <img src> can't send
 * it), or why that failed. Neither is returned for another path, so a frame never gets the
 * previous frame's image, whose URL is revoked once it is left.
 */
function useScreenshot(path: string | null): Omit<Screenshot, 'path'> {
  const [shot, setShot] = useState<Screenshot>();
  useEffect(() => {
    if (path === null) return;
    let objectUrl: string | undefined;
    let cancelled = false;
    void (async () => {
      try {
        const blob = await (await apiFetch(path)).blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setShot({ path, url: objectUrl });
      } catch (caught) {
        if (!cancelled) setShot({ path, error: errorText(caught) });
      }
    })();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);
  return shot?.path === path ? shot : {};
}

/** The frame's screenshot, cross-faded between steps, with a marker where the next action points. */
export function ScreenshotViewer({ episodeId, frame }: { episodeId: string; frame: Frame }) {
  // Natural size of the image that last loaded, by object URL: the marker needs this frame's.
  const [size, setSize] = useState<{ url: string; w: number; h: number } | null>(null);
  const { url, error } = useScreenshot(
    frame.has_screenshot ? `/episodes/${episodeId}/frames/${frame.index}/screenshot` : null,
  );
  const point = actionPoint(frame.action);

  if (!frame.has_screenshot) {
    return (
      <div className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-border-button-default text-body-2-regular text-text-tertiary">
        <RiImageLine aria-hidden className="size-5" />
        No screenshot at this step
      </div>
    );
  }

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl border border-border-button-default bg-background-secondary-default"
      style={{ aspectRatio: size ? `${size.w} / ${size.h}` : '4 / 3' }}
    >
      <AnimatePresence initial={false} mode="popLayout">
        {url && (
          // biome-ignore lint/performance/noImgElement: static export shows API screenshots as-is; next/image can't optimise them.
          <motion.img
            key={url}
            src={url}
            alt={`Screen at step ${frame.index}`}
            onLoad={(event) =>
              setSize({ url, w: event.currentTarget.naturalWidth, h: event.currentTarget.naturalHeight })
            }
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="absolute inset-0 block h-full w-full"
          />
        )}
      </AnimatePresence>
      {!url && (
        // Fades in late so quick loads while stepping don't flash it.
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className={cx(
            'absolute inset-0 flex items-center justify-center p-4 text-center text-body-2-regular',
            error ? 'text-text-error-primary' : 'text-text-tertiary',
          )}
        >
          {error ? `Could not load the screenshot: ${error}` : 'Loading screenshot'}
        </motion.p>
      )}
      {point && url && size?.url === url && (
        <span
          aria-hidden
          className="pointer-events-none absolute size-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-400 bg-accent-400/20 shadow-[0_0_0_4px_color-mix(in_srgb,var(--color-accent-400)_25%,transparent)]"
          style={{ left: `${(point.x / size.w) * 100}%`, top: `${(point.y / size.h) * 100}%` }}
        />
      )}
    </div>
  );
}
