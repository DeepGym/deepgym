import { readFile } from 'node:fs/promises';

export const dynamic = 'force-static';

// Serves deepgym/docs/rl-primer.html as is, so the site has no copy of it.
export async function GET() {
  const html = await readFile(`${process.cwd()}/../docs/rl-primer.html`);
  return new Response(html, { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
}
