import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import type { ReactNode } from 'react';
import { Provider } from '@/components/provider';
import { source } from '@/lib/source';
import './global.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = { title: { default: 'DeepGym', template: '%s | DeepGym' } };

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>
          <DocsLayout
            tree={source.getPageTree()}
            nav={{ title: 'DeepGym' }}
            githubUrl="https://github.com/DeepGym/deepgym"
          >
            {children}
          </DocsLayout>
        </Provider>
      </body>
    </html>
  );
}
