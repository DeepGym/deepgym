"""api_request scores what the solution fetches and prints, not what its source says."""

import pytest

from deepgym import DeepGym, load_environment

ENV = load_environment('api_request')
REFERENCE = ENV.verifier_path.with_name('reference_solution.py').read_text(encoding='utf-8')

HTTP_CLIENT = """
import http.client, json, sys, urllib.parse

url = urllib.parse.urlsplit(sys.argv[1])
conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
conn.request('GET', url.path)
print(json.loads(conn.getresponse().read())['slideshow']['title'])
"""

HARDCODED = """
# Fetch https://httpbin.org/json with urllib.request, parse the json response
# and print the slideshow title.
print('Sample Slide Show')
"""

KEYWORDS = """
import json, urllib.request  # requests httpx urlopen http.client .json()
# data['slideshow']['title']
print('title')
"""

FIRST_TITLE_IN_BODY = """
import re, sys, urllib.request

body = urllib.request.urlopen(sys.argv[1], timeout=10).read().decode()
print(re.search(r'"title": "([^"]*)"', body).group(1))
"""

FIRST_SLIDE_TITLE = REFERENCE.replace(
    "['slideshow']['title']", "['slideshow']['slides'][0]['title']"
)


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='local')


@pytest.mark.parametrize('solution', [REFERENCE, HTTP_CLIENT], ids=['urllib', 'http_client'])
def test_solutions_that_fetch_and_extract_score_full(dg: DeepGym, solution: str) -> None:
    result = dg.run(ENV, solution)

    assert result.score == 1.0, result.output


@pytest.mark.parametrize(
    'solution',
    [HARDCODED, KEYWORDS, FIRST_TITLE_IN_BODY, FIRST_SLIDE_TITLE],
    ids=['hardcoded_print', 'keyword_stuffing', 'first_title_in_body', 'wrong_field'],
)
def test_shortcuts_fail(dg: DeepGym, solution: str) -> None:
    result = dg.run(ENV, solution)

    assert (result.passed, result.score < 1.0) == (False, True), result.output
