"""Pyright, at zero.

Configuration lives in [tool.pyright] in pyproject.toml. `make typecheck`
runs the same check and prints the errors themselves.

Three of these are `# pyright: ignore` on a call that is correct at run time
and mistyped by the library: gensim annotates `prune_at` and `keep_n` as int
where None is what disables them, and docling's backend options carry
defaults pyright cannot see.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.types


@pytest.fixture(scope="module")
def report() -> dict:
    """Runs pyright over everything [tool.pyright] includes."""
    finished = subprocess.run(
        ["pyright", "--outputjson"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not finished.stdout.strip():
        pytest.fail(f"pyright produced no report: {finished.stderr[-2000:]}")
    return json.loads(finished.stdout)


def test_there_are_no_type_errors(report: dict) -> None:
    """Every error, named with the line it is on."""
    errors = [
        f"{Path(item['file']).resolve().relative_to(ROOT)}"
        f":{item['range']['start']['line'] + 1} "
        f"[{item.get('rule')}] {' '.join(item['message'].split())[:160]}"
        for item in report["generalDiagnostics"]
        if item["severity"] == "error"
    ]

    assert not errors, "\n".join(errors)


def test_the_whole_project_was_checked(report: dict) -> None:
    """An include path that stops matching would otherwise pass as clean."""
    assert report["summary"]["filesAnalyzed"] > 100, report["summary"]
