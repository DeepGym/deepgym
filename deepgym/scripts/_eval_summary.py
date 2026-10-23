"""Summaries of eval-script records: what an average may count, and what it may not."""

from __future__ import annotations

from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass rate and mean score over the runs that measure what was asked: judged runs of
    a whole answer from the requested source. A model reply cut off at an output limit,
    one of unknown completion (an old cache), a reference fallback or a run with no
    verdict is counted apart, never averaged in."""
    measured = [
        record
        for record in records
        if record['has_verdict']
        and record['complete'] is True
        and record.get('answer_source') != 'reference_fallback'
    ]
    return {
        'count': len(records),
        'measured': len(measured),
        'passed': sum(bool(record['passed']) for record in measured),
        'avg_score': round(sum(record['score'] for record in measured) / len(measured), 4)
        if measured
        else None,
        'truncated': sum(record['complete'] is False for record in records),
        'unknown_completion': sum(record['complete'] is None for record in records),
        'reference_fallback': sum(r.get('answer_source') == 'reference_fallback' for r in records),
        'no_verdict': sum(not record['has_verdict'] for record in records),
    }
