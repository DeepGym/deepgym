"""Verifier for synthetic CVE-2021-44228 log triage."""

import ast
import json
import re
import sys
from urllib.parse import unquote

EXPECTED_IPS = {'203.0.113.42', '198.51.100.77'}
EXPECTED_PAYLOADS = {'jndi:ldap', 'jndi:dns'}  # the two lookups in the log
ACTION_TERMS = {'patch', 'block', 'search', 'logs'}
# ponytail: keyword check, not language understanding. A clause (split at . ; : ! ?)
# with any negation earns no action terms, so "patch, but don't wait" loses credit
# rather than "patching is not needed" gaining it; use a judge model to do better.
_NEGATION = re.compile(r"\b(?:not|no|never|avoid|without)\b|n['’]t\b")


def _read_output(solution_path):
    text = open(solution_path, encoding='utf-8').read().strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        return parsed if isinstance(parsed, dict) else {}


def _string_set(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _affirmed_actions(text):
    found = set()
    for clause in re.split(r'[.;:!?](?:\s|$)|\n', text):
        if not _NEGATION.search(clause):
            found |= {term for term in ACTION_TERMS if term in clause}
    return found


def verify(solution_path, test_cases_path=None):
    # Only the answer's own fields count: keywords elsewhere in the output earn nothing.
    data = _read_output(solution_path)
    ips = _string_set(data.get('source_ips'))
    indicators = unquote(' '.join(_string_set(data.get('payload_indicators')))).lower()
    action_text = str(data.get('recommended_action', '')).lower()

    cases = []
    score = 0.0

    # Every IP that didn't probe (a benign one, or one not in the log) costs credit.
    ip_score = len(ips & EXPECTED_IPS) / len(ips | EXPECTED_IPS)
    score += 0.4 * ip_score
    cases.append({'id': 'source_ips', 'passed': ip_score == 1.0, 'score': ip_score})

    cve_ok = str(data.get('cve', '')).strip().lower() == 'cve-2021-44228'
    cve_score = 1.0 if cve_ok and 'log4shell' in str(data.get('vulnerability', '')).lower() else 0.0
    score += 0.2 * cve_score
    cases.append({'id': 'cve_identification', 'passed': cve_score == 1.0, 'score': cve_score})

    payload_hits = sum(1 for payload in EXPECTED_PAYLOADS if payload in indicators)
    payload_score = payload_hits / len(EXPECTED_PAYLOADS)
    score += 0.25 * payload_score
    cases.append(
        {'id': 'payload_indicators', 'passed': payload_score == 1.0, 'score': payload_score}
    )

    action_score = min(1.0, len(_affirmed_actions(action_text)) / 3)
    score += 0.15 * action_score
    cases.append(
        {'id': 'recommended_action', 'passed': action_score >= 0.66, 'score': action_score}
    )

    score = round(min(1.0, score), 4)
    return {
        'schema_version': '1.0',
        'score': score,
        'passed': score >= 0.95,
        'details': f'cyber log triage score={score:.2f}',
        'reward_components': {
            'source_ips': round(0.4 * ip_score, 4),
            'cve_identification': round(0.2 * cve_score, 4),
            'payload_indicators': round(0.25 * payload_score, 4),
            'recommended_action': round(0.15 * action_score, 4),
        },
        'cases': cases,
    }


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1])))
