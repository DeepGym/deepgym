/** DeepGym glyph: a stack of three offset bars (environments) with a check. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <rect x="3" y="4" width="14" height="3.5" rx="1.75" fill="currentColor" opacity="0.45" />
      <rect x="5" y="10.25" width="14" height="3.5" rx="1.75" fill="currentColor" opacity="0.7" />
      <rect x="7" y="16.5" width="14" height="3.5" rx="1.75" fill="currentColor" />
    </svg>
  );
}
