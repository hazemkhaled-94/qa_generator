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
    "question-worker",
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


@pytest.fixture(scope="module")
def orchestrated() -> dict:
    """The same, with the `orchestration` profile switched on.

    A second resolution rather than a flag on the first: `compose config`
    leaves a profile's services out entirely unless the profile is named,
    so the fixture above cannot see the two Dagster services at all, and
    the tests that read every service must not start seeing them either.
    """
    answered = compose("--profile", "orchestration", "config", "--format", "json")
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
        ("question-worker", "question_generation.run"),
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
    for worker in (
        "api",
        "parse-worker",
        "chunk-worker",
        "extract-worker",
        "question-worker",
    ):
        waits_for = set(services[worker].get("depends_on", {}))
        assert "postgres" in waits_for, f"{worker} waits for {sorted(waits_for)}"


#: The services built from the backend image. One image, one PYTHONPATH, and
#: one working tree bind-mounted over it.
BACKEND_SERVICES = (
    "api",
    "parse-worker",
    "chunk-worker",
    "extract-worker",
    "topic-worker",
    "question-worker",
)


def backend_packages() -> set[str]:
    """Every importable package under backend/, as the image lays them out."""
    return {
        found.name
        for found in (ROOT / "backend").iterdir()
        if found.is_dir() and (found / "__init__.py").exists()
    }


def mounted_packages(service: dict) -> set[str]:
    """Which backend packages one service bind-mounts over its image."""
    return {
        Path(volume["source"]).name
        for volume in service.get("volumes", [])
        if volume.get("type") == "bind" and "/backend/" in str(volume.get("source", ""))
    }


@pytest.mark.parametrize("service", BACKEND_SERVICES)
def test_every_backend_service_mounts_every_backend_package(resolved, service) -> None:
    """One image serves all six, so one of them mounting less runs a mix.

    The failure this catches is not a missing file. The package is in the
    image, so a container whose mount list has fallen behind runs today's
    source for the packages it mounts and the image's for the rest, and the
    symptom is a route that 404s or a function that is two versions old.

    The api is the one that gets forgotten, because it is the only backend
    service that does not take the `x-worker` anchor: it has ports, a
    healthcheck and a volume list of its own.
    """
    missing = backend_packages() - mounted_packages(resolved["services"][service])

    assert not missing, (
        f"{service} does not mount {', '.join(sorted(missing))}. It runs the "
        f"image's copy of those and the working tree's copy of the rest."
    )


def test_the_backend_services_all_mount_the_same_packages(resolved) -> None:
    """Whichever list is right, they have to agree on it.

    Two lists that differ is the shape of the bug above; this says so
    without having to know which of them is correct.
    """
    mounts = {
        service: mounted_packages(resolved["services"][service])
        for service in BACKEND_SERVICES
    }
    odd = {name: sorted(held) for name, held in mounts.items() if held != mounts["api"]}

    assert not odd, f"api mounts {sorted(mounts['api'])}, but {odd}"


def test_the_scan_finds_the_packages_it_is_meant_to_guard(resolved) -> None:
    """A refactor that moved the packages would otherwise pass silently."""
    found = backend_packages()

    assert {"api", "question_generation", "extraction"} <= found, sorted(found)
    assert mounted_packages(resolved["services"]["api"]), "no bind mounts were read"


#: The orchestrator's two processes. Behind the `orchestration` profile, so
#: `compose up` does not start them; `compose config` lists them anyway,
#: which is what lets them be read here.
DAGSTER_SERVICES = ("dagster-webserver", "dagster-daemon")


@pytest.mark.parametrize("service", DAGSTER_SERVICES)
def test_the_orchestrator_cannot_reach_the_application_tables(
    orchestrated, service
) -> None:
    """It decides when a stage runs; it never runs one.

    The queue has three faces already - the Start button, the make targets
    and the stage routes - and all three go through the same repository. A
    fourth that reached past them into the tables would be a second way of
    moving a row, and would disagree with the other three the first time
    one of them changed.

    Holding no credential is what makes that structural rather than a
    convention somebody remembers.
    """
    environment = orchestrated["services"][service].get("environment", {})
    reachable = sorted(
        name
        for name in ("DATABASE_URL", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY")
        if environment.get(name)
    )

    assert not reachable, (
        f"{service} is given {', '.join(reachable)}. The orchestrator posts "
        f"to the stage routes and reads /status back, and nothing else."
    )


@pytest.mark.parametrize("service", DAGSTER_SERVICES)
def test_the_orchestrator_is_told_where_the_backend_is(orchestrated, service) -> None:
    """The orchestrator is told where the backend is.

    The one address it holds. Without it the code location loads and
    every asset fails on its first call.
    """
    assert orchestrated["services"][service].get("environment", {}).get("BACKEND_URL")


@pytest.mark.parametrize("service", DAGSTER_SERVICES)
def test_the_orchestrator_does_not_run_the_backend_image(orchestrated, service) -> None:
    """The orchestrator is not built from the backend image.

    That image serves the api and all five workers and carries torch,
    spaCy, Docling and litellm because some process in it needs each. This
    one makes HTTP calls: sharing it would put three gigabytes behind a
    process whose whole job is to POST and poll.
    """
    service_image = orchestrated["services"][service].get("image", "")

    assert service_image != orchestrated["services"]["api"].get("image"), service_image
