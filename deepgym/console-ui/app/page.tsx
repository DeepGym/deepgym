import { Suspense } from 'react';
import { OverviewPage } from '@/components/overview/overview-page';

export default function Page() {
  return (
    <Suspense>
      <OverviewPage />
    </Suspense>
  );
}
