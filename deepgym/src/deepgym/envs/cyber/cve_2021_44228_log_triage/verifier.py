"""Verifier for synthetic CVE-2021-44228 log triage."""

import ast
import json
import re
import sys

EXPECTED_IPS = {'203.0.113.42', '198.51.100.77'}
BENIGN_IPS = {'198.51.100.10', '192.0.2.55'}
REQUIRED_TERMS = {'cve-2021-44228', 'log4shell'}
REQUIRED_PAYLOAD_TERMS = {'jndi', 'ldap', 'dns'}


def _read_output(solution_path):
    text = open(solution_path, encoding='utf-8').read().strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), text
            except json.JSONDecodeError:
                pass
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        return parsed if isinstance(parsed, dict) else {}, text


def _string_set(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def verify(solution_path, test_cases_path=None):
    data, raw = _read_output(solution_path)
    raw_lower = raw.lower()
    ips = _string_set(data.get('source_ips'))
    indicators = ' '.join(_string_set(data.get('payload_indicators'))).lower()
    cve_text = f"{data.get('cve', '')} {data.get('vulnerability', '')}".lower()
    action_text = str(data.get('recommended_action', '')).lower()

    cases = []
    score = 0.0

    ip_score = len(ips & EXPECTED_IPS) / len(EXPECTED_IPS)
    if ips & BENIGN_IPS:
        ip_score = max(0.0, ip_score - 0.5)
    score += 0.4 * ip_score
    cases.append({'id': 'source_ips', 'passed': ip_score == 1.0, 'score': ip_score})

    cve_score = (
        1.0
        if all(term in cve_text or term in raw_lower for term in REQUIRED_TERMS)
        else 0.0
    )
    score += 0.2 * cve_score
    cases.append({'id': 'cve_identification', 'passed': cve_score == 1.0, 'score': cve_score})

    payload_hits = sum(
        1 for term in REQUIRED_PAYLOAD_TERMS if term in indicators or term in raw_lower
    )
    payload_score = payload_hits / len(REQUIRED_PAYLOAD_TERMS)
    score += 0.25 * payload_score
    cases.append(
        {'id': 'payload_indicators', 'passed': payload_score == 1.0, 'score': payload_score}
    )

    action_terms = {'patch', 'block', 'search', 'logs'}
    action_score = min(1.0, sum(1 for term in action_terms if term in action_text) / 3)
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
