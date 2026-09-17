"""When the orchestrator decides the pipeline should start.

The one piece here that acts on its own. Everything else in this package
runs because a person asked or a clock fired; this runs because something
was uploaded, so what it declines to do matters as much as what it does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dagster", reason="the pipeline group is not installed")

from dagster import RunRequest, SkipReason, build_sensor_context

import orchestration
from orchestration.client import Queue


@pytest.fixture(autouse=True)
def settings(monkeypatch) -> None:
    """The environment the code location reads, so no .env is needed."""
    monkeypatch.setenv("BACKEND_URL", "http://api:8000")
    monkeypatch.setenv("ORCHESTRATION_DRAIN_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("ORCHESTRATION_POLL_SECONDS", "1")


def answering(monkeypatch, **rows: int) -> None:
    """Makes /parsing/status report these counts."""

    class Stub:
        """Stands in for the backend client."""

        def __init__(self, *args, **kwargs) -> None:
            """Accepts whatever the sensor passes."""

        def status(self, stage: str) -> Queue:
            """Reports the queue the test asked for."""
            return Queue(
                stage=stage,
                working=rows.get("in_progress", 0) > 0,
                rows={name: count for name, count in rows.items() if count},
            )

    monkeypatch.setattr(orchestration, "Backend", Stub)


def evaluate():
    """Runs one sensor tick."""
    return orchestration.arrivals(build_sensor_context())


def test_a_document_nobody_asked_for_starts_a_run(monkeypatch) -> None:
    """Which is the whole reason the sensor exists."""
    answering(monkeypatch, new=3)

    assert isinstance(evaluate(), RunRequest)


def test_an_empty_queue_starts_nothing(monkeypatch) -> None:
    """A tick a minute over an idle stack must not queue a run a minute."""
    answering(monkeypatch, parsed=8)

    answer = evaluate()
    assert isinstance(answer, SkipReason)
    assert "waiting" in str(answer.skip_message)


def test_nothing_starts_while_a_run_is_already_working(monkeypatch) -> None:
    """`new` beside `pending` is an upload that arrived mid-run.

    Starting a second run for it would have two runs driving one queue,
    both watching the same rows and each crediting the other's work.
    """
    answering(monkeypatch, new=2, pending=5)

    answer = evaluate()
    assert isinstance(answer, SkipReason)
    assert "already queued" in str(answer.skip_message)


def test_two_ticks_over_an_unchanged_queue_ask_for_one_run(monkeypatch) -> None:
    """The run key is what Dagster deduplicates on.

    Without it the sensor asks again every minute until a worker picks
    the documents up, which on a stage with no worker is for ever.
    """
    answering(monkeypatch, new=3)

    assert evaluate().run_key == evaluate().run_key


def test_a_later_upload_asks_for_its_own_run(monkeypatch) -> None:
    """An upload during a run gets a run of its own.

    A run that has already started will not pick up what arrived after
    it, so the key has to change when the queue does.
    """
    answering(monkeypatch, new=3)
    first = evaluate().run_key
    answering(monkeypatch, new=4)

    assert evaluate().run_key != first
