"""Computer-use environment support for DeepGym.

Enable environments where agents interact with browsers, desktops, and GUIs
inside Daytona sandboxes with Docker-in-Docker support.

Verifiers for computer-use tasks can check:
- Screenshot comparison (pixel diff, structural similarity)
- DOM state (CSS selectors, element presence/absence)
- URL/navigation state
- File system changes
- Process/application state
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from deepgym.models import Environment


class ComputerUseEnvironment(Environment):
    """Define a computer-use task environment.

    Extend Environment so that core.py's dg.run() can access verifier_path,
    test_cases, env_vars, and snapshot fields. The agent receives a task
    (e.g. 'Navigate to github.com and star the DeepGym repo') and interacts
    with a browser/desktop via tool calls. The verifier checks the final state
    (screenshot, DOM, URL, files, etc.).
    """

    type: Literal['computer-use'] = 'computer-use'
    """Override environment type."""

    setup_script: str = ''
    """Shell script to set up the environment (install browser, open URL, etc.)."""

    tools: list[str] = Field(
        default_factory=lambda: ['screenshot', 'click', 'type', 'scroll', 'bash']
    )
    """Available tools for the agent."""

    viewport: dict[str, int] = Field(default_factory=lambda: {'width': 1280, 'height': 720})
    """Browser viewport dimensions."""

    timeout: int = Field(default=120, ge=1)
    """Max time in seconds for the agent to complete the task."""

    snapshot: str | None = 'docker-dind'
    """Daytona snapshot with browser support. docker-dind enables running browsers in sandbox."""


class ScreenshotVerifier:
    """Verify computer-use tasks by checking screenshots.

    Use pixel comparison and structural similarity to check if
    the agent achieved the desired visual state.
    """

    @staticmethod
    def compare_screenshots(actual_path: str, expected_path: str, threshold: float = 0.85) -> dict:
        """Compare two screenshots and return similarity score.

        Use mean squared error for pixel comparison and histogram correlation
        for structural similarity. Require only Pillow (PIL).

        Args:
            actual_path: Path to the actual screenshot.
            expected_path: Path to the expected screenshot.
            threshold: Minimum similarity score to pass (0.0-1.0).

        Returns:
            Dict with 'score', 'passed', and 'details' keys.
        """
        try:
            from PIL import Image
        except ImportError:
            return {
                'score': 0.0,
                'passed': False,
                'details': 'Pillow is required for screenshot comparison. '
                'Install with: pip install Pillow',
            }

        try:
            actual = Image.open(actual_path).convert('RGB')
            expected = Image.open(expected_path).convert('RGB')
        except (FileNotFoundError, OSError) as exc:
            return {
                'score': 0.0,
                'passed': False,
                'details': f'Failed to open image: {exc}',
            }

        # Resize actual to match expected dimensions if they differ.
        if actual.size != expected.size:
            actual = actual.resize(expected.size, Image.LANCZOS)

        # Compute mean squared error (MSE) over all pixels.
        actual_pixels = list(actual.getdata())
        expected_pixels = list(expected.getdata())
        total_pixels = len(expected_pixels)

        if total_pixels == 0:
            return {'score': 0.0, 'passed': False, 'details': 'Empty image'}

        mse = sum(
            sum((a - b) ** 2 for a, b in zip(ap, ep))
            for ap, ep in zip(actual_pixels, expected_pixels)
        ) / (total_pixels * 3)

        # Normalize MSE to a 0-1 similarity score.
        # Max possible MSE is 255^2 = 65025.
        pixel_similarity = 1.0 - min(mse / 65025.0, 1.0)

        # Histogram correlation for structural similarity.
        actual_hist = actual.histogram()
        expected_hist = expected.histogram()
        hist_sum = sum(a * b for a, b in zip(actual_hist, expected_hist))
        hist_norm_a = sum(a * a for a in actual_hist) ** 0.5
        hist_norm_b = sum(b * b for b in expected_hist) ** 0.5

        if hist_norm_a > 0 and hist_norm_b > 0:
            hist_similarity = hist_sum / (hist_norm_a * hist_norm_b)
        else:
            hist_similarity = 0.0

        # Combined score: weighted average (70% pixel, 30% histogram).
        score = 0.7 * pixel_similarity + 0.3 * hist_similarity
        score = max(0.0, min(1.0, score))
        passed = score >= threshold

        return {
            'score': round(score, 4),
            'passed': passed,
            'details': (
                f'pixel_similarity={pixel_similarity:.4f}, '
                f'hist_similarity={hist_similarity:.4f}, '
                f'combined={score:.4f}, threshold={threshold}'
            ),
        }

    @staticmethod
    def check_dom_element(html: str, selector: str, expected_text: str | None = None) -> dict:
        """Check if a DOM element exists and optionally matches text.

        Use simple string matching for MVP (no full HTML parser needed).
        Support id selectors (#id), class selectors (.class), and tag selectors.

        Args:
            html: Raw HTML string to search.
            selector: CSS-like selector (#id, .class, or tag name).
            expected_text: Optional text content to match within the element.

        Returns:
            Dict with 'found', 'passed', and 'details' keys.
        """
        found = False
        matched_text = False

        if selector.startswith('#'):
            # ID selector: look for id="value" or id='value'.
            id_val = selector[1:]
            pattern = rf'id\s*=\s*["\']?{re.escape(id_val)}["\']?'
            found = bool(re.search(pattern, html, re.IGNORECASE))
        elif selector.startswith('.'):
            # Class selector: look for class="... value ...".
            class_val = selector[1:]
            pattern = rf'class\s*=\s*["\'][^"\']*\b{re.escape(class_val)}\b[^"\']*["\']'
            found = bool(re.search(pattern, html, re.IGNORECASE))
        else:
            # Tag selector: look for <tag ...> or <tag>.
            pattern = rf'<{re.escape(selector)}[\s>/]'
            found = bool(re.search(pattern, html, re.IGNORECASE))

        if found and expected_text is not None:
            matched_text = expected_text in html
        elif found:
            matched_text = True

        passed = found and matched_text

        return {
            'found': found,
            'passed': passed,
            'details': (f'selector={selector!r}, found={found}, text_match={matched_text}'),
        }

    @staticmethod
    def check_url(current_url: str, expected_pattern: str) -> dict:
        """Check if current URL matches expected pattern (support regex).

        Args:
            current_url: The actual URL to check.
            expected_pattern: Regex pattern the URL should match.

        Returns:
            Dict with 'matched', 'passed', and 'details' keys.
        """
        try:
            matched = bool(re.search(expected_pattern, current_url))
        except re.error as exc:
            return {
                'matched': False,
                'passed': False,
                'details': f'Invalid regex pattern: {exc}',
            }

        return {
            'matched': matched,
            'passed': matched,
            'details': (f'url={current_url!r}, pattern={expected_pattern!r}, matched={matched}'),
        }


class ToolUseEnvironment(Environment):
    """Define a tool-use task environment.

    Extend Environment so that core.py's dg.run() can access verifier_path,
    test_cases, env_vars, and snapshot fields. The agent receives a task and
    a set of tools (APIs, CLI commands, file operations). The verifier checks
    the outcome (API responses, file state, CLI output).
    """

    type: Literal['tool-use'] = 'tool-use'
    """Override environment type."""

    tools: list[str] = Field(default_factory=list)
    """Available tools: bash, http, file_read, file_write, etc."""

    setup_script: str = ''
    """Script to set up mock APIs, files, etc."""

    expected_state: dict | None = None
    """Expected final state to check against."""

    timeout: int = Field(default=60, ge=1)
    """Max time in seconds for the agent to complete the task."""
