'use client';

import {
  SegmentedControl,
  SegmentedControlItem,
} from '@/components/base/segmented-control/segmented-control';
import { DEFAULT_RANGE, parseRange, RANGES, type Range } from '@/lib/stats';
import { useQueryParam } from '@/lib/use-query-param';

/** The stats time range kept in the URL (?range=24h|7d|30d; the default 7d leaves it out). */
export function useRange(): [Range, (range: Range) => void] {
  const [value, setValue] = useQueryParam('range');
  return [parseRange(value), (range) => setValue(range === DEFAULT_RANGE ? null : range)];
}

/** 24h / 7d / 30d switch for the window the stats cover. */
export function RangeControl({ value, onChange }: { value: Range; onChange: (range: Range) => void }) {
  return (
    <SegmentedControl
      aria-label="Time range"
      selectedKeys={[value]}
      onSelectionChange={(keys) => onChange(parseRange(String([...keys][0])))}
    >
      {Object.keys(RANGES).map((key) => (
        <SegmentedControlItem key={key} id={key}>
          {key}
        </SegmentedControlItem>
      ))}
    </SegmentedControl>
  );
}
