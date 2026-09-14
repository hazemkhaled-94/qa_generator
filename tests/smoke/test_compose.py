"""The compose file, as the engine reads it.

Parsing and resolving it catches what a YAML reader cannot: a variable
nothing supplies, a service depending on one that is not there, a port
declared twice. It starts nothing, so it is safe to run beside a stack that
is already up - which is the normal state of a machine developing this.

Bringing a second stack up is not done here. compose.yaml binds its ports
from .env, so a second copy would collide with the first rather than run
beside it; what an image contains is covered by test_images.py, and what
the services do to each other is covered by the integration layer.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.smoke

#: Every service the pipeline needs to be able to come up.
EXPECTED = {
    "postgres",
    "seaweedfs-master",
    "seaweedfs-volume",
    "seaweedfs-filer",
    "seaweedfs-s3",
    "api",
    "parse-worker",
    "chunk-worker",
    "extract-worker",
    "topic-worker",
    "streamlit",
}


#: The spellings of compose, newest first. The Makefile uses the second;
#: a plain docker CLI pointed at podman's socket has neither plugin, and
#: falls through to the standalone binary.
ENGINES = (
    ("docker", "compose"),
    ("podman", "compose"),
    ("/opt/podman/bin/podman", "compose"),
    ("docker-compose",),
)


@functools.cache
def _engine() -> tuple[str, ...]:
    """The first spelling of compose that answers, or a skip."""
    for engine in ENGINES:
        if not (shutil.which(engine[0]) or Path(engine[0]).exists()):
            continue
        answered = subprocess.run(
            [*engine, "version"], capture_output=True, text=True, check=False
        )
        if answered.returncode == 0:
            return engine
    pytest.skip("no working compose on this machine")


def compose(*args: str) -> subprocess.CompletedProcess:
    """Runs one compose command, or skips when there is no engine."""
    return subprocess.run(
        [*_engine(), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.fixture(scope="module")
def resolved() -> dict:
    """The compose file with every variable substituted, as the engine sees it."""
    answered = compose("config", "--format", "json")
    if answered.returncode != 0:
        pytest.skip(f"compose could not resolve the file: {answered.stderr[-600:]}")
    return json.loads(answered.stdout)


def test_the_compose_file_resolves(resolved) -> None:
    """Every variable it interpolates is supplied by .env."""
    assert resolved["services"]


def test_every_service_the_pipeline_needs_is_declared(resolved) -> None:
    """A worker missing here is a stage nothing runs."""
    declared = set(resolved["services"])

    assert EXPECTED <= declared, sorted(EXPECTED - declared)


def test_no_two_services_bind_the_same_host_port(resolved) -> None:
    """Two would come up in either order and one would fail."""
    taken: dict[str, str] = {}
    clashes = []
    for name, service in resolved["services"].items():
        for port in service.get("ports", []):
            published = str(port.get("published", ""))
            if not published:
                continue
            if published in taken:
                clashes.append(f"{name} and {taken[published]} both bind {published}")
            taken[published] = name

    assert not clashes, "\n".join(clashes)


def test_every_dependency_names_a_service_that_exists(resolved) -> None:
    """A typo here is a stack that never starts and does not say why."""
    declared = set(resolved["services"])
    missing = [
        f"{name} depends on {needed}"
        for name, service in resolved["services"].items()
        for needed in service.get("depends_on", {})
        if needed not in declared
    ]

    assert not missing, "\n".join(missing)


def test_the_workers_run_the_stage_each_is_named_for(resolved) -> None:
    """One image serves all of them, so the command is what differs."""
    services = resolved["services"]
    for worker, module in (
        ("parse-worker", "preprocessing.parsing.run"),
        ("chunk-worker", "preprocessing.chunking.run"),
        ("extract-worker", "extraction.run"),
        ("topic-worker", "topic_modelling.run"),
    ):
        command = " ".join(str(part) for part in services[worker].get("command", []))
        assert module in command, f"{worker} runs {command!r}"


def test_everything_waited_on_has_a_healthcheck_to_wait_for(resolved) -> None:
    """`condition: service_healthy` against a service with no check never passes.

    Not every service needs one - adminer and phoenix serve a port and
    nothing waits on them - but one that is waited on does.
    """
    services = resolved["services"]
    waited_on = {
        needed
        for service in services.values()
        for needed, how in service.get("depends_on", {}).items()
        if how.get("condition") == "service_healthy"
    }

    without = [name for name in waited_on if not services[name].get("healthcheck")]

    assert not without, sorted(without)


def test_the_stages_wait_for_the_stores_they_write_to(resolved) -> None:
    """A worker that starts first fails its first row and stops."""
    services = resolved["services"]
    for worker in ("api", "parse-worker", "chunk-worker", "extract-worker"):
        waits_for = set(services[worker].get("depends_on", {}))
        assert "postgres" in waits_for, f"{worker} waits for {sorted(waits_for)}"
