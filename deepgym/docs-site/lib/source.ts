import { loader } from 'fumadocs-core/source';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import { defineDocs } from 'fumadocs-mdx/macro';

// The site renders deepgym/docs in place. Those files have no frontmatter: each page's
// title comes from its first `# ` heading.
const docs = defineDocs({
  dir: '../docs',
  docs: { schema: pageSchema.partial({ title: true }) },
  meta: { schema: metaSchema },
});

export const source = loader({
  baseUrl: '/',
  source: docs.toFumadocsSource(),
});
