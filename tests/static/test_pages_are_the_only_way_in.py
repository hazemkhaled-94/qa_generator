"""The page object is the frontend layer's only handle on a running view.

`tests/frontend/pages.py` holds the selectors, so a test says what a reader
does and what they see, and a layout change is one edit there. That only
holds while every test goes through it. Twenty-eight assertions did not:
they reached `view.app.number_input` and `view.app.get("download_button")`
straight through the page object to Streamlit's own test API, which put a
widget's spelling back in the test bodies and made the page object a thing
you could route around.

So: no test module touches `.app`. The page object does, once, and that is
the point of it. When a page needs something the object cannot say yet, the
fix is a method on the object rather than a reach-through here.

The same rule for `conftest`: a test module that says `from conftest import`
is importing whichever conftest.py pytest happened to load first under that
bare name, which is a different file depending on the collection order. Four
modules did, and `pytest tests/unit tests/frontend` failed to collect all
four of them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "tests/frontend"

#: The page object itself, which is where the reaching-through belongs.
ALLOWED = {"pages.py", "conftest.py"}

TESTS = sorted(path for path in FRONTEND.glob("test_*.py") if path.name not in ALLOWED)


def _reaches_through(source: str) -> list[str]:
    """Every `<something>.app.<name>` one module reads."""
    return [
        f"{node.value.attr}.{node.attr}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "app"
    ]


@pytest.mark.parametrize("path", TESTS, ids=lambda path: path.name)
def test_a_page_test_works_the_page_object_and_not_the_app(path: Path) -> None:
    """A widget's spelling belongs in pages.py, in one place."""
    reached = _reaches_through(path.read_text())

    assert not reached, (
        f"{path.name} reaches past the page object to Streamlit: "
        f"{sorted(set(reached))}. Add the accessor to pages.View instead."
    )


@pytest.mark.parametrize("path", TESTS, ids=lambda path: path.name)
def test_a_test_module_does_not_import_a_conftest_by_name(path: Path) -> None:
    """Which conftest that names depends on what ran first."""
    imported = [
        node.module
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module == "conftest"
    ]

    assert not imported, (
        f"{path.name} imports from `conftest`, which resolves to whichever "
        f"conftest.py pytest loaded first. Put the name in pages.py."
    )
