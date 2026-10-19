// Fail the build when a link in the exported site points at a page or heading that does not exist.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const out = join(import.meta.dirname, '..', 'out');
const pages = readdirSync(out, { recursive: true, encoding: 'utf8' }).filter((f) => f.endsWith('.html'));
const html = new Map(pages.map((f) => [f, readFileSync(join(out, f), 'utf8')]));
const ids = new Map(
  Array.from(html, ([f, s]) => [f, new Set(Array.from(s.matchAll(/\sid="([^"]+)"/g), (m) => m[1]))]),
);

// Map a URL path to the exported file that serves it, as a static host would.
function fileFor(pathname: string): string | undefined {
  const rel = decodeURIComponent(pathname).replace(/^\//, '');
  const candidates = rel === '' || rel.endsWith('/') ? [`${rel}index.html`] : [rel, `${rel}/index.html`];
  return candidates.find((c) => existsSync(join(out, c)));
}

const broken: string[] = [];
for (const [file, content] of html) {
  const base = new URL(`/${file.replace(/index\.html$/, '')}`, 'https://site.invalid');
  for (const [, raw] of content.matchAll(/<a\s[^>]*?href="([^"]*)"/g)) {
    const href = raw.replaceAll('&amp;', '&');
    const url = new URL(href, base);
    if (url.origin !== base.origin) continue;
    const target = fileFor(url.pathname);
    const hash = decodeURIComponent(url.hash.slice(1));
    const targetIds = target && ids.get(target);
    if (!target) broken.push(`${file}: ${href} (no page)`);
    else if (hash && targetIds && !targetIds.has(hash)) broken.push(`${file}: ${href} (no #${hash})`);
  }
}

if (broken.length) {
  console.error(`${broken.length} broken links:\n${broken.join('\n')}`);
  process.exit(1);
}
console.log(`Links OK in ${html.size} pages.`);
