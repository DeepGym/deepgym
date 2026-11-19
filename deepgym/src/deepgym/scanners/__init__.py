"""Trace scanners: find what the verifier didn't catch in recorded traces, and cite it.

``scan_traces(traces, scanners)`` runs scanners such as :class:`PatternScanner` (regexes),
:class:`ProtectedPathScanner` (actions that change protected paths) and
:class:`LLMScanner` (a judge model given an instruction) and reports their findings and
errors; ``group_findings`` clusters findings, ``finding_labels`` stores them as label
marks, and ``scanner_agreement`` measures each scanner against human labels.
"""

from deepgym.scanners.core import (
    SCAN_ERROR,
    Finding,
    FindingGroup,
    HumanMark,
    ScanError,
    Scanner,
    ScannerAgreement,
    ScannerInfo,
    ScannerSource,
    ScanReport,
    Severity,
    field_text,
    finding_labels,
    group_findings,
    scan_traces,
    scanner_agreement,
)
from deepgym.scanners.llm import SCAN_PROMPT, SCAN_PROMPT_VERSION, LLMScanner
from deepgym.scanners.rules import (
    Pattern,
    PatternScanner,
    ProtectedPathScanner,
    RuleScanner,
    ScanField,
    load_rule_scanners,
)

__all__ = [
    'SCAN_ERROR',
    'SCAN_PROMPT',
    'SCAN_PROMPT_VERSION',
    'Finding',
    'FindingGroup',
    'HumanMark',
    'LLMScanner',
    'Pattern',
    'PatternScanner',
    'ProtectedPathScanner',
    'RuleScanner',
    'ScanError',
    'ScanField',
    'ScanReport',
    'Scanner',
    'ScannerAgreement',
    'ScannerInfo',
    'ScannerSource',
    'Severity',
    'field_text',
    'finding_labels',
    'group_findings',
    'load_rule_scanners',
    'scan_traces',
    'scanner_agreement',
]
