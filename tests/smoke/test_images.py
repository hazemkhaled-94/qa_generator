"""The images, built and looked inside.

The in-process layers import from the working tree, so they cannot see what
an image does or does not contain. This is the layer that can: a dependency
declared in requirements.txt and missing from the lock produces a green
build and a container that fails on its first import, which is how the
topic worker once shipped without gensim.

Slow, and excluded from `make test`. `make test-smoke` runs it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.smoke

#: A distribution name at the start of a requirement line, before any
#: extra or version range.
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")

BACKEND = "qa_generator-smoke-backend"
FRONTEND = "qa_generator-smoke-frontend"

#: Every module a container's entry point reaches. One import each is what
#: separates an image that starts from one that does not.
ENTRY_POINTS = (
    "api.main",
    "ingestion.run",
    "preprocessing.parsing.run",
    "preprocessing.chunking.run",
    "extraction.run",
    "topic_modelling.run",
    "stages.cli",
    "telemetry",
)


def container() -> str:
    """The engine to build and run with, or a skip."""
    for name in ("docker", "podman"):
        found = shutil.which(name)
        if found:
            return found
    pytest.skip("no container engine on PATH")


def run(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Runs one engine command from the repository root."""
    return subprocess.run(
        [container(), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="module")
def backend_image() -> str:
    """Builds the backend image, under a tag of this suite's own.

    Its own tag so a run here never replaces the image a running stack is
    using.
    """
    built = run(
        "build",
        "-t",
        BACKEND,
        "-f",
        "backend/api/Dockerfile",
        ".",
    )
    assert built.returncode == 0, built.stderr[-4000:]
    return BACKEND


@pytest.fixture(scope="module")
def frontend_image() -> str:
    """Builds the frontend image, under a tag of this suite's own."""
    built = run("build", "-t", FRONTEND, "-f", "frontend/Dockerfile", ".")
    assert built.returncode == 0, built.stderr[-4000:]
    return FRONTEND


def test_the_backend_image_builds(backend_image) -> None:
    """Which is also what checks the lock pins everything declared."""
    assert backend_image


def test_every_entry_point_imports_inside_the_image(backend_image) -> None:
    """One image serves the api and all four workers.

    Under the tuning file the services read, because a stage reads its
    settings as it is imported and there are no defaults in code.
    """
    script = "; ".join(f"import {module}" for module in ENTRY_POINTS)
    imported = run(
        "run",
        "--rm",
        "--env-file",
        "configs/env/backend.env",
        "-e",
        "DATABASE_URL=postgresql+psycopg://unused:unused@localhost/x",
        "-e",
        "S3_ENDPOINT=http://unused:8333",
        "-e",
        "S3_ACCESS_KEY=unused",
        "-e",
        "S3_SECRET_KEY=unused",
        "--entrypoint",
        "python",
        backend_image,
        "-c",
        f"{script}; print('imported')",
        timeout=300,
    )

    assert "imported" in imported.stdout, imported.stderr[-4000:]


def test_the_spacy_pipelines_are_baked_in(backend_image) -> None:
    """The runtime has no network: a missing pipeline fails every passage."""
    loaded = run(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        backend_image,
        "-c",
        "import spacy; "
        "[spacy.load(m) for m in ('de_core_news_md', 'en_core_web_md')]; "
        "print('loaded')",
        timeout=300,
    )

    assert "loaded" in loaded.stdout, loaded.stderr[-4000:]


def test_the_image_carries_no_cuda_wheel(backend_image) -> None:
    """CPU-only torch, which is most of the difference in image size."""
    listed = run(
        "run",
        "--rm",
        "--entrypoint",
        "pip",
        backend_image,
        "list",
        "--format=json",
        timeout=300,
    )

    installed = {one["name"].lower() for one in json.loads(listed.stdout)}
    assert not [name for name in installed if name.startswith("nvidia-")], sorted(
        name for name in installed if name.startswith("nvidia-")
    )


def test_the_image_installs_everything_the_lock_pins(backend_image) -> None:
    """A lock entry that did not install is a container that starts and fails."""
    listed = run(
        "run",
        "--rm",
        "--entrypoint",
        "pip",
        backend_image,
        "list",
        "--format=json",
        timeout=300,
    )
    installed = {
        one["name"].lower().replace("_", "-") for one in json.loads(listed.stdout)
    }

    declared = {
        found.group(1).lower().replace("_", "-")
        for line in (ROOT / "backend/api/requirements.txt").read_text().splitlines()
        if (found := _NAME.match(line.split("#")[0].strip()))
    }

    assert declared <= installed, sorted(declared - installed)


def test_the_image_runs_as_a_user_that_is_not_root(backend_image) -> None:
    """Nothing in the pipeline needs root, and the volume is owned by app."""
    whoami = run("run", "--rm", "--entrypoint", "whoami", backend_image, timeout=120)

    assert whoami.stdout.strip() == "app", whoami.stdout


def test_the_frontend_image_builds_and_imports(frontend_image) -> None:
    """The frontend holds one URL and the telemetry package."""
    imported = run(
        "run",
        "--rm",
        "-e",
        "BACKEND_URL=http://backend.invalid",
        "-e",
        "PAGE_SIZE=25",
        "--entrypoint",
        "python",
        frontend_image,
        "-c",
        "import lib.config, lib.backend, telemetry; print('imported')",
        timeout=300,
    )

    assert "imported" in imported.stdout, imported.stderr[-4000:]
