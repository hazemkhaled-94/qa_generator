"""Every relative link in a README points at a file that is there.

Twenty-six READMEs point at each other and at the code they describe, and a
moved module breaks them silently: nothing imports a link, so nothing fails.
`backend/nlp/embedding.py` used to be `question_generation/embedding.py` and
three pages went on naming the old path.

Only relative targets are checked. An http link is somebody else's uptime
and a bare `#anchor` is the renderer's.
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


def _pages() -> list[Path]:
    """Every markdown file this repository maintains."""
    found = []
    for name in DOCUMENTED:
        one = ROOT / name
        found.extend([one] if one.is_file() else sorted(one.rglob("*.md")))
    return [p for p in found if "__pycache__" not in p.parts]


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(page: Path) -> None:
    """Each relative target of one page exists on disk."""
    broken = [
        target
        for target in _LINK.findall(page.read_text())
        if not target.startswith(("http://", "https://", "mailto:", "#"))
        # `[text](url)` is markdown ABOUT markdown in the chunking README.
        and target != "url"
        and not (page.parent / target.split("#")[0]).exists()
    ]
    assert not broken, f"{page.relative_to(ROOT)} links to {', '.join(broken)}"
