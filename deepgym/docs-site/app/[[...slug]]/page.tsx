import { DocsBody, DocsPage } from 'fumadocs-ui/layouts/docs/page';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import type { ComponentProps } from 'react';
import { source } from '@/lib/source';

export default async function Page(props: PageProps<'/[[...slug]]'>) {
  const page = source.getPage((await props.params).slug);
  if (!page) notFound();
  const MDX = page.data.body;

  // The docs link to each other as on GitHub (`sessions.md`, `../traces.md`). Fumadocs resolves
  // only hrefs that start with `./` or `../`, so a bare relative path gets `./` first.
  const a = ({ href, ...rest }: ComponentProps<'a'>) => {
    const relative = href && !/^([a-z][a-z\d+.-]*:|[/#.])/i.test(href) ? `./${href}` : href;
    return <defaultMdxComponents.a href={relative && source.resolveHref(relative, page)} {...rest} />;
  };

  // The markdown's own `# ` heading is the page title, so there is no DocsTitle and the
  // table of contents starts below it.
  return (
    <DocsPage toc={page.data.toc.filter((item) => item.depth > 1)}>
      <DocsBody>
        <MDX components={{ ...defaultMdxComponents, a }} />
      </DocsBody>
    </DocsPage>
  );
}

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: PageProps<'/[[...slug]]'>): Promise<Metadata> {
  const page = source.getPage((await props.params).slug);
  if (!page) notFound();
  return { title: page.data.title };
}
