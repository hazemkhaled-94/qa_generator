"""Pyright, held to a baseline rather than to zero.

The type errors already here are not this suite's to fix, and a gate that is
red on arrival gates nothing. So the count per file is recorded in
pyright_baseline.json and only a rise fails: new code is checked from the
start, and the existing errors can come down a file at a time.

Fixing some? Regenerate the baseline downwards:

    poetry run pyright --outputjson | poetry run python -c '...'

or just edit the number. A file that reaches zero comes out of the file.
"""

from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).parent / "pyright_baseline.json"

pytestmark = pytest.mark.types


@pytest.fixture(scope="module")
def errors() -> dict[str, int]:
    """Runs pyright and counts its errors per file."""
    finished = subprocess.run(
        ["pyright", "--outputjson"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not finished.stdout.strip():
        pytest.fail(f"pyright produced no report: {finished.stderr[-2000:]}")

    reported = json.loads(finished.stdout)["generalDiagnostics"]
    return collections.Counter(
        str(Path(item["file"]).resolve().relative_to(ROOT))
        for item in reported
        if item["severity"] == "error"
    )


@pytest.fixture(scope="module")
def baseline() -> dict[str, int]:
    """The recorded count per file."""
    return json.loads(BASELINE.read_text())


def test_no_file_has_more_type_errors_than_it_did(errors, baseline) -> None:
    """New code is type-checked from the start."""
    worse = {
        path: (count, baseline.get(path, 0))
        for path, count in errors.items()
        if count > baseline.get(path, 0)
    }

    assert not worse, "\n".join(
        f"{path}: {now} type error(s), was {before}. Run `make typecheck`."
        for path, (now, before) in sorted(worse.items())
    )


def test_the_baseline_names_no_file_that_is_already_clean(errors, baseline) -> None:
    """A file that has been fixed comes out, so the gate keeps tightening."""
    clean = sorted(path for path, count in baseline.items() if not errors.get(path))

    assert not clean, (
        f"these are clean now and must be removed from {BASELINE.name}: "
        f"{', '.join(clean)}"
    )
