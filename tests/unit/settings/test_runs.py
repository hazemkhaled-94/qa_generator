"""Which of three things gets to name the run a row belongs to.

The order matters more than any one of them. A worker is long-lived and
drains whatever arrives, so naming a run after the PROCESS - which is all
there used to be - named the container and not the work: a Dagster run, a
Start button and `make extract` all produced rows stamped with the same
worker, and nothing in the data said which of them had asked.
"""

from __future__ import annotations

import pytest

from settings import runs


@pytest.fixture(autouse=True)
def unnamed(monkeypatch):
    """A process nobody named, which is the ordinary deployment.

    Put back afterwards as well as before: the trigger lives in a
    ContextVar the whole process shares, so a test that left one set would
    name the lines of every test after it.
    """
    monkeypatch.delenv(runs.VARIABLE, raising=False)
    runs.triggered_by(None)
    yield
    runs.triggered_by(None)


def test_a_process_nobody_named_and_nothing_asked_for_names_itself() -> None:
    """The fallback, and the only thing there used to be."""
    assert runs.run_id()
    assert runs.run_id() == runs.run_id()


def test_the_run_that_asked_for_the_row_is_the_one_recorded() -> None:
    """The whole point: the id on a produced row is the id that asked.

    `StageQueue._claim` sets this from the row it took, so everything the
    worker writes while it holds that row belongs to whoever queued it.
    """
    alone = runs.run_id()
    runs.triggered_by("dagster-run-7")

    assert runs.run_id() == "dagster-run-7"
    assert runs.run_id() != alone


def test_a_row_nobody_asked_for_falls_back_to_the_process() -> None:
    """A claim always sets this, including to None.

    A row queued before any of this existed must not inherit the trigger
    of the row the worker held before it.
    """
    runs.triggered_by("dagster-run-7")

    runs.triggered_by(None)

    assert runs.run_id() != "dagster-run-7"


def test_a_run_somebody_named_outranks_the_one_that_asked(monkeypatch) -> None:
    """RUN_ID is a deliberate act and has to win.

    The A/B in `evaluation/README.md` is two drains under two names, and it
    would otherwise be renamed by whichever surface happened to press Start.
    """
    monkeypatch.setenv(runs.VARIABLE, "b-gemma4-12b")
    runs.triggered_by("dagster-run-7")

    assert runs.run_id() == "b-gemma4-12b"
