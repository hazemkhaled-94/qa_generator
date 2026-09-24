"""Every relative link in a README points at something that is there.

Twenty-six READMEs point at each other and at the code they describe, and a
moved module breaks them silently: nothing imports a link, so nothing fails.
`backend/nlp/embedding.py` used to be `question_generation/embedding.py` and
three pages went on naming the old path.

A heading a link names is checked too, when the link names one in another
markdown file. That is the half a path check misses: `docs/architecture.md`
pointed at `#a-worker-proves-the-model-before-it-claims-anything` for four
words of a heading that ends `...-and-waits-for-it`, so the file resolved
and the jump did not.

Only relative targets are checked. An http link is somebody else's uptime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where the project's own documentation lives. Not a glob over the whole
#: tree: a virtualenv and a cache both hold markdown nobody here wrote.
DOCUMENTED = (
    "README.md",
    "docs",
    "backend",
    "frontend",
    "telemetry",
    "orchestration",
    "review",
    "evaluation",
    "configs",
    "tests",
)

_LINK = re.compile(r"\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _pages() -> list[Path]:
    """Every markdown file this repository maintains."""
    found = []
    for name in DOCUMENTED:
        one = ROOT / name
        found.extend([one] if one.is_file() else sorted(one.rglob("*.md")))
    return [p for p in found if "__pycache__" not in p.parts]


def _targets(page: Path) -> list[str]:
    """The relative links one page makes.

    `[text](url)` is markdown ABOUT markdown in the chunking README, and
    the only literal that looks like a path and is not one.
    """
    return [
        target
        for target in _LINK.findall(page.read_text())
        if not target.startswith(("http://", "https://", "mailto:")) and target != "url"
    ]


def _anchors(page: Path) -> set[str]:
    """The fragments GitHub will resolve in one page, from its headings.

    Lower-cased, punctuation dropped, spaces hyphenated. An em dash is
    dropped rather than hyphenated, which is why `Platform — the encoders`
    is `platform--the-encoders` and not `platform-the-encoders`.
    """
    found = set()
    for heading in _HEADING.findall(page.read_text()):
        kept = re.sub(r"[^\w\s-]", "", heading.strip().lower())
        found.add(re.sub(r"\s+", "-", kept.strip()))
    return found


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(page: Path) -> None:
    """Each relative target of one page exists on disk."""
    broken = [
        target
        for target in _targets(page)
        if not target.startswith("#")
        and not (page.parent / target.split("#")[0]).exists()
    ]
    assert not broken, f"{page.relative_to(ROOT)} links to {', '.join(broken)}"


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_heading_a_link_names_is_there(page: Path) -> None:
    """Each `#fragment` names a heading the page it points at carries."""
    broken = []
    for target in _targets(page):
        path, _, anchor = target.partition("#")
        if not anchor:
            continue
        at = page if not path else page.parent / path
        if at.suffix == ".md" and at.exists() and anchor not in _anchors(at):
            broken.append(target)
    assert not broken, f"{page.relative_to(ROOT)} names {', '.join(broken)}"
