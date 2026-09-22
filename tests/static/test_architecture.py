"""The architecture page, against the contracts it describes.

docs/architecture.md draws the layer graph that .importlinter enforces. A
diagram is prose: renaming a layer, adding one or dropping one changes what
`make lint-imports` checks and changes nothing on the page, and a diagram
that is wrong about the import graph is worse than no diagram, because it
is the thing somebody reads before deciding where a module goes.

Only the names are checked here, not the edges. The order of the layers is
the contract's to state and `make lint-imports` already fails on it; what
this catches is a layer the page has never heard of.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / ".importlinter"
PAGE = ROOT / "docs" / "architecture.md"

#: A layers contract writes alternatives on one line, separated by pipes:
#: `extraction | ingestion | preprocessing`.
_ALTERNATIVES = re.compile(r"\s*\|\s*")


def _config() -> configparser.ConfigParser:
    """Reads .importlinter, which is an ini file."""
    parsed = configparser.ConfigParser()
    parsed.read(CONTRACTS, encoding="utf-8")
    return parsed


def _layers() -> set[str]:
    """Every module named in the layers contract, alternatives split out."""
    declared = _config()["importlinter:contract:layers"]["layers"]
    found: set[str] = set()
    for line in declared.splitlines():
        found.update(part for part in _ALTERNATIVES.split(line.strip()) if part)
    return found


def _names() -> set[str]:
    """The title of every contract, which the page quotes as a table row."""
    parsed = _config()
    return {
        parsed[section]["name"]
        for section in parsed.sections()
        if section.startswith("importlinter:contract:")
    }


@pytest.fixture(scope="module")
def page() -> str:
    """The architecture page, read once."""
    return PAGE.read_text(encoding="utf-8")


@pytest.mark.parametrize("layer", sorted(_layers()))
def test_every_layer_is_on_the_page(layer: str, page: str) -> None:
    """A module the contract layers is a module the diagram draws."""
    assert layer in page, (
        f"{layer} is a layer in .importlinter and is named nowhere in "
        f"docs/architecture.md. Add it to the layer graph, or take it out "
        f"of the contract."
    )


@pytest.mark.parametrize("name", sorted(_names()))
def test_every_contract_is_on_the_page(name: str, page: str) -> None:
    """The page states what each contract says, in the contract's words."""
    assert name in page, (
        f"docs/architecture.md does not quote the contract named {name!r}. "
        f"A renamed contract is a renamed rule, and the page states the rules."
    )
