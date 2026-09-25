"""The code location, as Dagster loads it.

Loading is most of the test. A code location that does not import is a
Dagster UI showing one red box and no assets, and the way that usually
happens is a typo in a dependency name - which resolves to an asset key
that exists nowhere and fails at load, not at run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dagster", reason="the pipeline group is not installed")

from dagster import AssetKey, Definitions

from orchestration import arrivals, corpus, defs, nightly

#: The order a document moves through the pipeline. Each waits for the one
#: before it, which is what makes the graph a pipeline rather than six
#: things that happen to be in a list.
#:
#: `assessments` is last and is not a stage the corpus moves THROUGH: it
#: judges what the five before it produced and writes nothing any of them
#: reads. It is in the chain so that it runs after all of them, which is
#: the one thing about its position that matters.
ORDER = (
    "parsed_documents",
    "passages",
    "facts",
    "topics",
    "questions",
    "assessments",
)

SPECS = {spec.key.to_user_string(): spec for spec in defs.resolve_all_asset_specs()}


def test_the_code_location_loads() -> None:
    """Dagster's own check that everything in it resolves.

    This is most of the value of the file. A code location that does not
    load is a UI showing one red box and no assets at all.
    """
    Definitions.validate_loadable(defs)


def test_every_stage_is_an_asset() -> None:
    """A stage missing here is a stage the orchestrator cannot run."""
    assert set(SPECS) == set(ORDER)


@pytest.mark.parametrize(
    ("stage", "waits_for"), list(zip(ORDER[1:], ORDER[:-1], strict=True))
)
def test_each_stage_waits_for_the_one_before_it(stage: str, waits_for: str) -> None:
    """Each stage depends on the one before it.

    The chain is the point: extraction before a fit, a fit before
    questions. Wired the other way round, a run would write questions
    about the topics from last time.
    """
    assert {dep.asset_key for dep in SPECS[stage].deps} == {AssetKey(waits_for)}


def test_the_first_stage_waits_for_nothing() -> None:
    """Parsing is fed by uploads, which are not an asset."""
    assert not SPECS[ORDER[0]].deps


def test_every_stage_has_a_check_for_its_failed_rows() -> None:
    """A failed row is not a failed run, so it is reported and not raised."""
    checked = {
        key.asset_key.to_user_string()
        for key in defs.resolve_asset_graph().asset_check_keys
    }

    assert checked == set(ORDER)


def test_the_schedule_and_the_sensor_ship_stopped() -> None:
    """Neither trigger fires until somebody switches it on.

    A stack that starts running the pipeline the moment it comes up is
    one nobody chose. Both are enabled in the UI, deliberately.
    """
    assert nightly.default_status.value == "STOPPED", nightly.default_status
    assert arrivals.default_status.value == "STOPPED", arrivals.default_status


def test_the_nightly_job_is_the_whole_corpus() -> None:
    """A refit is corpus-wide and goes stale on every new document."""
    assert nightly.job.name == corpus.name
    assert nightly.cron_schedule == "0 2 * * *"
