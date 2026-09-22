"""The two architecture rules the root README states, checked.

Configuration lives in `.importlinter` at the repository root, which is also
where the reasoning is. `make lint-imports` runs the same check and prints
the offending edges itself.

`PYTHONPATH=backend` because the image puts each backend package at the top
level: `extraction` is the name that resolves in a container, not
`backend.extraction`, and a contract written against the other name would
pass by matching nothing.

This does not replace `test_api_stays_light.py`. That one reads imports as
syntax so it can tell an import that RUNS from one deferred into a function
body, and the api's weight rests entirely on that distinction - the topic
service defers pyLDAvis, `extraction/run.py` defers the embedder. grimp,
which import-linter builds its graph with, counts both alike.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: How many edges the graph should hold. A contract over a graph that
#: collapsed - a renamed package, a path that stopped resolving - passes by
#: having nothing to check, which is the failure this number exists for.
#: Measured at 431 when the contracts were written.
LEAST_DEPENDENCIES = 300


@pytest.fixture(scope="module")
def report() -> str:
    """Runs import-linter over the contracts, however it exits."""
    if shutil.which("lint-imports") is None:
        pytest.skip("import-linter is not installed")
    finished = subprocess.run(
        ["lint-imports", "--verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'backend'}{os.pathsep}{ROOT}"},
    )
    return finished.stdout + finished.stderr


def test_every_contract_is_kept(report: str) -> None:
    """Broken contracts, with the edge that broke each."""
    assert "Contracts: " in report, f"import-linter did not run:\n{report[-2000:]}"
    assert " 0 broken" in report, report[-4000:]


def test_the_graph_is_the_whole_backend(report: str) -> None:
    """A contract over a graph that found nothing passes for the wrong reason."""
    found = [line for line in report.splitlines() if "dependencies." in line]
    assert found, f"import-linter reported no graph size:\n{report[-2000:]}"

    counted = int(found[0].split("files,")[1].split("dependencies")[0].strip())
    assert counted >= LEAST_DEPENDENCIES, (
        f"the import graph holds {counted} dependencies, against "
        f"{LEAST_DEPENDENCIES} when the contracts were written. A package "
        f"renamed out of `root_packages` in .importlinter leaves its "
        f"contracts passing over nothing."
    )
