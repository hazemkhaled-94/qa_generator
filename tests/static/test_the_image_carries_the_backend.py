"""Every first-party backend module is in the image and under the mounts.

The Dockerfile names what it copies one line per directory, and compose
mounts the same directories over it so an edit is live without a rebuild.
Both are lists somebody maintains by hand, and a module added to `backend/`
that is not added to both is a container that builds, starts and dies on
`ModuleNotFoundError` at import.

Which is what happened. `backend/confidence.py` is the one first-party
module that is a file rather than a package, so a line per directory did
not carry it and neither list gained one. The api and all five workers
died on `No module named 'confidence'` and the UI went blank.

`tests/smoke/test_images.py` would have caught it - it imports every entry
point inside the built image - but smoke builds two images and runs
nightly, never on a pull request. This reads the two files instead and
takes milliseconds, which is what makes it a gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
DOCKERFILE = ROOT / "backend/api/Dockerfile"
COMPOSE = ROOT / "compose.yaml"

#: Copied into no image on purpose: compose mounts it read-only into the
#: api and the workers, and the archive CLI is run through that mount
#: rather than baked in. Named here so the exception is a decision.
MOUNTED_ONLY = {"archive"}


def _first_party() -> set[str]:
    """Every name `import <name>` reaches inside the image.

    A package is a directory holding `__init__.py`; a module is a `.py`
    file beside them. Both are top-level names on the image's path, which
    is why both have to be carried.
    """
    found = set()
    for path in BACKEND.iterdir():
        if path.name.startswith((".", "_")) or path.name == "__pycache__":
            continue
        if path.is_dir() and (path / "__init__.py").exists():
            found.add(path.name)
        elif path.suffix == ".py":
            found.add(path.stem)
    return found


def _copied() -> set[str]:
    """Every top-level backend name the Dockerfile copies.

    Only the top level: the build also copies `backend/api/requirements.lock`
    into the layer that installs it, and a file inside a package is not a
    name anything imports.
    """
    copied = set()
    for line in DOCKERFILE.read_text().splitlines():
        parts = line.split()
        if not (len(parts) >= 2 and parts[0] == "COPY"):
            continue
        for argument in parts[1:-1]:
            named = argument.removeprefix("backend/").rstrip("/")
            if argument.startswith("backend/") and "/" not in named:
                copied.add(named.removesuffix(".py"))
    return copied


def _mounted() -> set[str]:
    """Every backend name compose mounts over the image."""
    mounted = set()
    for line in COMPOSE.read_text().splitlines():
        stripped = line.strip().lstrip("- ")
        if stripped.startswith("./backend/"):
            source = stripped.split(":")[0]
            mounted.add(Path(source).name.removesuffix(".py"))
    return mounted


def test_there_is_something_to_check() -> None:
    """A parser that silently matched nothing would pass every test below."""
    assert len(_first_party()) > 10, sorted(_first_party())
    assert len(_copied()) > 10, sorted(_copied())
    assert len(_mounted()) > 10, sorted(_mounted())


@pytest.mark.parametrize("name", sorted(_first_party() - MOUNTED_ONLY))
def test_the_image_carries_every_backend_module(name: str) -> None:
    """A module the image does not hold is an import error at start-up."""
    assert name in _copied(), (
        f"backend/{name} is importable but backend/api/Dockerfile copies no "
        f"such path. Add a COPY line for it."
    )


@pytest.mark.parametrize("name", sorted(_first_party()))
def test_compose_mounts_every_backend_module(name: str) -> None:
    """Otherwise an edit to it needs a rebuild and every sibling does not."""
    assert name in _mounted(), (
        f"backend/{name} is importable but compose.yaml mounts no such path, "
        f"so editing it does nothing until the image is rebuilt."
    )


def test_nothing_is_copied_that_is_not_there() -> None:
    """A COPY of a path that has been renamed fails the build, late."""
    assert _copied() - _first_party() == set(), (
        f"the Dockerfile copies what backend/ no longer holds: "
        f"{sorted(_copied() - _first_party())}"
    )
