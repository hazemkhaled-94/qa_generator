"""Every requirement a service declares is pinned in the lock it installs.

The same comparison backend/api/Dockerfile makes, run without a build: only
the lock is installed, so a dependency added to requirements.txt alone
produces a green image that fails on its first import.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The directories holding a requirements.txt and the lock built from it.
SERVICES = (ROOT / "backend/api", ROOT / "frontend")

_DECLARED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")


def _folded(name: str) -> str:
    """Folds a distribution name to PEP 503 form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(path: Path, pattern: re.Pattern[str]) -> set[str]:
    """Reads the distribution names a file lists, folded."""
    return {
        _folded(found.group(1))
        for line in path.read_text().splitlines()
        if (found := pattern.match(line.split("#")[0].strip()))
    }


@pytest.mark.parametrize("service", SERVICES, ids=lambda path: path.name)
def test_the_lock_pins_every_declared_requirement(service: Path) -> None:
    """The lock names everything requirements.txt does."""
    declared = _names(service / "requirements.txt", _DECLARED)
    assert declared, f"{service.name}/requirements.txt declares nothing"

    missing = declared - _names(service / "requirements.lock", _PINNED)
    assert not missing, (
        f"{service.name}/requirements.lock does not pin {', '.join(sorted(missing))}. "
        f"Run `make lock`."
    )
