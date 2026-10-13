import type { Metadata } from 'next';
import { Suspense } from 'react';
import { EnvironmentsPage } from '@/components/environments/environments-page';

export const metadata: Metadata = { title: 'Environments - DeepGym Console' };

export default function Page() {
  return (
    <Suspense>
      <EnvironmentsPage />
    </Suspense>
  );
}
