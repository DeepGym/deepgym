'use client';

import { RiPlayLine } from '@remixicon/react';
import { useState } from 'react';
import { Button } from '@/components/base/buttons/button';
import { Label } from '@/components/base/input/label';
import {
  SegmentedControl,
  SegmentedControlItem,
} from '@/components/base/segmented-control/segmented-control';
import { Select, SelectItem } from '@/components/base/select/select';
import { Textarea } from '@/components/base/textarea/textarea';
import { Modal } from '@/components/ui/overlay';
import { api, errorText, useApi } from '@/lib/api';
import { formatDuration } from '@/lib/format';
import { type EnvironmentSummary, RUN_MODES, type RunMode, type RunSummary } from '@/lib/types';
import { EnvPicker } from './env-picker';
import { KEEPS_GOING, useTypicalDuration } from './run-progress';

interface NewRunDialogProps {
  open: boolean;
  /** Environment to preselect, e.g. from an environment's "New run" button. */
  initialEnv: string | null;
  onClose: () => void;
  onCreated: (runs: RunSummary[]) => void;
}

function NewRunForm({ initialEnv, onClose, onCreated }: Omit<NewRunDialogProps, 'open'>) {
  const { data } = useApi<EnvironmentSummary[]>('/environments?kind=single-turn');
  const { data: config } = useApi<{ local_custom_code: boolean }>('/runs/config');
  const [selected, setSelected] = useState(() => new Set(initialEnv ? [initialEnv] : []));
  const [mode, setMode] = useState<RunMode>('local');
  const [source, setSource] = useState<'reference' | 'custom'>('reference');
  const [solution, setSolution] = useState('');
  const [error, setError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const typical = useTypicalDuration(selected.size === 1 ? [...selected][0] : null);

  // Custom code may only run on this machine when the console was started with --allow-local-exec.
  const localBlocked = source === 'custom' && config?.local_custom_code === false;
  const ready =
    selected.size > 0 &&
    (source === 'reference' || solution.trim() !== '') &&
    !(localBlocked && mode === 'local');
  const submit = async () => {
    setSubmitting(true);
    setError(undefined);
    try {
      const body = { environments: [...selected], mode, solution: source === 'custom' ? solution : null };
      onCreated(await api<RunSummary[]>('/runs', { method: 'POST', body: JSON.stringify(body) }));
      onClose();
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      className="flex flex-col gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (ready) void submit();
      }}
    >
      <EnvPicker envs={data ?? []} selected={selected} onChange={setSelected} />
      <div className="grid gap-5 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <Label>Provider</Label>
          <Select
            aria-label="Provider"
            selectedKey={mode}
            onSelectionChange={(key) => setMode(key as RunMode)}
          >
            {RUN_MODES.map((value) => (
              <SelectItem key={value} id={value}>
                {value}
              </SelectItem>
            ))}
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <Label>Solution</Label>
          <SegmentedControl
            aria-label="Solution"
            selectedKeys={[source]}
            onSelectionChange={(keys) => setSource([...keys][0] as 'reference' | 'custom')}
          >
            <SegmentedControlItem id="reference">Reference</SegmentedControlItem>
            <SegmentedControlItem id="custom">Paste code</SegmentedControlItem>
          </SegmentedControl>
        </div>
      </div>
      {source === 'custom' && (
        <Textarea
          label="Solution code"
          hint="The same code runs against every selected environment."
          placeholder="def solve(...):"
          rows={10}
          value={solution}
          onChange={setSolution}
          inputClassName="font-mono text-body-2-regular"
        />
      )}
      {localBlocked && (mode === 'local' || mode === 'auto') && (
        <p className="rounded-2lg bg-status-yellow-background px-3 py-2 text-body-2-medium text-status-yellow-text">
          Local runs of custom code are disabled. Pick docker, apple, daytona, modal or e2b, or restart the
          console with <code className="font-mono">--allow-local-exec</code>.
          {mode === 'auto' && ' Auto runs locally unless DAYTONA_API_KEY is set.'}
        </p>
      )}
      {error && (
        <p
          role="alert"
          className="rounded-2lg bg-background-tertiary-error px-3 py-2 text-body-2-medium text-text-error-primary"
        >
          {error}
        </p>
      )}
      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-separator-border pt-4">
        <p className="me-auto min-w-0 flex-1 basis-60 text-caption-1-medium text-text-tertiary">
          {KEEPS_GOING}
          {typical !== null && ` ${[...selected][0]} usually takes about ${formatDuration(typical)}.`}
        </p>
        <Button variant="secondary" onClick={onClose}>
          Cancel
        </Button>
        <Button type="submit" leadingIcon={RiPlayLine} disabled={!ready || submitting}>
          {submitting ? 'Starting' : `Start ${selected.size || ''} run${selected.size === 1 ? '' : 's'}`}
        </Button>
      </div>
    </form>
  );
}

/** Modal to start runs: environments, provider, and the reference or a pasted solution. */
export function NewRunDialog({ open, ...props }: NewRunDialogProps) {
  return (
    <Modal
      isOpen={open}
      onClose={props.onClose}
      title="New run"
      description="One run per environment; several environments make a small eval."
    >
      {open && <NewRunForm {...props} />}
    </Modal>
  );
}
