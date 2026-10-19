import { createMDX } from 'fumadocs-mdx/next';
import type { NextConfig } from 'next';

const config: NextConfig = {
  output: 'export',
  trailingSlash: true,
  // The content lives in ../docs, outside this package, so the bundler root is deepgym/.
  turbopack: { root: `${import.meta.dirname}/..` },
};

export default createMDX()(config);
