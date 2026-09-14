"""The CI workflows, read as the files they are.

A workflow only reports a mistake when it runs, and some mistakes are
quiet: a marker renamed here and not there deselects every test and the
job goes green having run nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.yml"))

#: Markers a workflow selects on, which must be markers pytest knows.
_SELECTOR = re.compile(r'-m\s+"([^"]+)"')


def loaded(path: Path) -> dict:
    """One workflow, parsed."""
    return yaml.safe_load(path.read_text())


def test_there_are_workflows_to_check() -> None:
    """A moved directory would otherwise pass as nothing to say."""
    assert WORKFLOWS, "no workflows in .github/workflows"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_parses_and_declares_jobs(path: Path) -> None:
    """Valid YAML, with jobs in it."""
    workflow = loaded(path)

    assert workflow.get("jobs"), path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_local_action_a_job_uses_exists(path: Path) -> None:
    """`uses: ./...` is a path, and a renamed one fails every job at once."""
    missing = []
    for job in loaded(path)["jobs"].values():
        for step in job.get("steps", []):
            used = step.get("uses", "")
            if used.startswith("./") and not (ROOT / used / "action.yml").exists():
                missing.append(used)

    assert not missing, sorted(missing)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_marker_a_workflow_selects_on_is_registered(path: Path) -> None:
    """A renamed marker would deselect everything and pass."""
    import tomllib

    declared = {
        line.split(":")[0]
        for line in tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
            "pytest"
        ]["ini_options"]["markers"]
    }

    selected = {
        word
        for expression in _SELECTOR.findall(path.read_text())
        for word in re.findall(r"[a-z_]+", expression)
        if word not in ("not", "and", "or")
    }

    assert selected, f"{path.name} selects on no marker at all"
    assert selected <= declared, sorted(selected - declared)


def test_the_gate_waits_for_every_other_job() -> None:
    """One job for a branch protection rule to require.

    So that adding a job above does not also mean editing the rule.
    """
    jobs = loaded(ROOT / ".github/workflows/ci.yml")["jobs"]
    gate = jobs["gate"]

    assert set(gate["needs"]) == set(jobs) - {"gate"}, gate["needs"]


def test_the_gate_names_every_job_it_waits_for_in_its_condition() -> None:
    """A job added to `needs` and not to the check passes unexamined."""
    jobs = loaded(ROOT / ".github/workflows/ci.yml")["jobs"]
    reported = " ".join(step.get("run", "") for step in jobs["gate"]["steps"])

    for needed in jobs["gate"]["needs"]:
        assert f"needs.{needed}.result" in reported, needed


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_that_sets_up_the_project_checks_out_first(path: Path) -> None:
    """The composite action is in the repository it is setting up."""
    for name, job in loaded(path)["jobs"].items():
        steps = [step.get("uses", "") for step in job.get("steps", [])]
        if "./.github/actions/setup" not in steps:
            continue
        assert any(one.startswith("actions/checkout") for one in steps), name
        assert steps.index("./.github/actions/setup") > next(
            index
            for index, one in enumerate(steps)
            if one.startswith("actions/checkout")
        ), name
