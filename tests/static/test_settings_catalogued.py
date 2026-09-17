"""Every setting the code reads is described, and every description is read.

`test_settings_documented` checks that a setting is declared in a file a
deployment edits. This checks the other half: that it is also in
`settings.catalog`, which is what a page draws a control from and what a
route refuses a bad value with.

Both directions matter. A setting the code reads and the catalogue does not
describe cannot be configured through the API at all, which is the whole
point of the catalogue. A setting the catalogue describes and nothing reads
is a control that does nothing - worse than a missing one, because it looks
like it works.

Read as syntax rather than by importing, the way the sibling scan is, so a
failure names the setting rather than needing an environment to load one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from settings.catalog import BY_NAME, SETTINGS, Service, of

ROOT = Path(__file__).resolve().parents[2]

#: The readers in backend/settings/env.py, by name. `mapping` is here and is
#: not in the sibling scan, which is why a mix could be read without being
#: declared anywhere.
#:
#: The three leading-underscore names are a stage's own readers, which take
#: the setting's name first and hand it to `mapping` or `csv` through a
#: variable. Without them the mixes and the answer bounds are invisible to a
#: scan that only looks at the readers themselves.
READERS = frozenset(
    {
        "required",
        "optional",
        "integer",
        "decimal",
        "boolean",
        "csv",
        "mapping",
        "_kinds",
        "_weights",
        "_bounds",
    }
)

#: Settings read through a variable rather than by name, so no scan of the
#: syntax can find them. The pool sizes are built into the engine's URL in
#: database/qa_generator/engine.py.
READ_INDIRECTLY = {"DATABASE_POOL_SIZE", "DATABASE_POOL_OVERFLOW"}

#: Directories holding the code that runs in a container.
SOURCES = ("backend", "frontend", "telemetry")

#: Settings the code reads that no page configures. Each is read before a
#: service could ask a database for it, or belongs to the frontend process
#: rather than to the pipeline.
UNCONFIGURABLE = {
    # Read to reach the store that would hold the settings.
    "DATABASE_URL",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    # Read before a service has read its settings.
    "LOG_LEVEL",
    "LOG_DIR",
    # The frontend's own, read in its process and not the backend's.
    "BACKEND_URL",
    "PAGE_SIZE",
}


def _read_by(source: str) -> set[str]:
    """Finds the settings one module reads, by literal name."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and node.args:
            called = node.func
            name = (
                called.id
                if isinstance(called, ast.Name)
                else getattr(called, "attr", None)
            )
            first = node.args[0]
            if (
                name in READERS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                found.add(first.value)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value.isupper()
            and "environ" in ast.dump(node.value)
        ):
            found.add(node.slice.value)
    return found


def _every_name() -> set[str]:
    """Every setting any module in the image reads, by literal name."""
    found: set[str] = set()
    for source in SOURCES:
        for path in sorted((ROOT / source).glob("**/*.py")):
            if "__pycache__" in path.parts or path.name == "catalog.py":
                continue
            found |= _read_by(path.read_text())
    return found


READS = _every_name()


def test_the_scan_finds_the_settings_it_is_meant_to_guard() -> None:
    """A refactor that moved the readers would otherwise pass silently."""
    assert "NLP_MODELS" in READS, sorted(READS)
    assert "QUESTIONS_TYPE_MIX" in READS, "the mapping reader is scanned here"
    assert len(READS) > 50, sorted(READS)


@pytest.mark.parametrize("name", sorted(READS))
def test_every_setting_the_code_reads_is_described(name: str) -> None:
    """Named one at a time, so a failure says which setting is undescribed."""
    if name in UNCONFIGURABLE:
        pytest.skip("read before a service could ask for it, or not the backend's")

    assert name in BY_NAME, (
        f"{name} is read by the code but not described in settings/catalog.py, "
        f"so no page can configure it and no route can refuse a bad value for "
        f"it. Add a Setting for it, or name it in UNCONFIGURABLE here and say "
        f"why."
    )


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda one: one.name)
def test_every_setting_described_is_one_the_code_reads(setting) -> None:
    """A control that does nothing looks like one that works."""
    if setting.name in READ_INDIRECTLY:
        pytest.skip("read through a variable, so the syntax does not name it")

    assert setting.name in READS, (
        f"{setting.name} is described in settings/catalog.py but nothing reads "
        f"it, so configuring it would do nothing at all."
    )


def test_no_setting_is_described_twice() -> None:
    """Two descriptions of one setting are two controls for it."""
    names = [one.name for one in SETTINGS]

    assert len(names) == len(set(names)), sorted(
        name for name in names if names.count(name) > 1
    )


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda one: one.name)
def test_every_setting_says_what_it_does(setting) -> None:
    """The line a page shows beside the control."""
    assert setting.help.strip(), setting.name
    assert setting.help.rstrip().endswith("."), (
        f"{setting.name}: the help is shown as a sentence, so it ends like one"
    )


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda one: one.name)
def test_bounds_are_the_right_way_round(setting) -> None:
    """A low above a high accepts nothing at all."""
    if setting.low is not None and setting.high is not None:
        assert setting.low <= setting.high, setting.name


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda one: one.name)
def test_only_a_number_carries_bounds(setting) -> None:
    """Bounds on a flag or a list would never be checked."""
    if setting.kind not in ("integer", "decimal"):
        assert setting.low is None and setting.high is None, setting.name


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda one: one.name)
def test_what_a_setting_stales_is_a_stage(setting) -> None:
    """A page offers the stage's own rerun as the remedy, so it must exist."""
    stages = ("parsing", "chunking", "extraction", "topics", "questions")

    assert all(one in stages for one in setting.invalidates), setting.name


def test_every_service_configures_something() -> None:
    """A page with nothing to configure should not offer a panel."""
    for service in Service.__args__:
        assert of(service), f"{service} configures nothing"


def test_every_setting_belongs_to_a_service_that_exists() -> None:
    """The service name is what a page looks itself up by."""
    for one in SETTINGS:
        assert one.service in Service.__args__, one.name


_DECLARATION = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def test_every_setting_described_is_also_declared_for_a_deployment() -> None:
    """The catalogue describes it; a file a deployment edits still supplies it.

    The environment stays required: the store overrides a setting, it never
    supplies one, so a setting missing from both files would stop a service
    at start-up however well the catalogue described it.
    """
    declared = {
        name
        for file in ("configs/env/backend.env", ".env.example")
        for name in _DECLARATION.findall((ROOT / file).read_text())
    }
    undeclared = {one.name for one in SETTINGS} - declared

    assert not undeclared, (
        f"{', '.join(sorted(undeclared))} is described in the catalogue but "
        f"declared in neither configs/env/backend.env nor .env.example"
    )
