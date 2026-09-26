"""The Pipeline panel on Documents, run against a scripted backend.

The one control in this frontend that is not a stage's. What it covers is
the half no integration test reaches: which buttons a given corpus offers,
what each of them sends, and what the panel does when the deployment has no
orchestrator to ask - which is a supported way to run this and must not read
as a broken page.
"""

from __future__ import annotations

import pytest
from pages import Answers, orchestration, pipeline

pytestmark = pytest.mark.frontend

#: The view this module is about. `page` in conftest.py reads it.
VIEW = "documents"


def client(**replaced) -> Answers:
    """A pipeline client answering a corpus a test describes."""
    return Answers(
        state=pipeline(**replaced),
        run=(True, "Run abc12345 started."),
        act=lambda action: (True, f"{action} happened."),
        automate=(True, "Saved."),
    )


# ── What the panel offers ─────────────────────────────────────────────────


def test_the_panel_is_drawn_on_the_page_a_document_lands_on(page) -> None:
    """After an upload, "now take it through" is asked here."""
    opened = page()

    assert "Pipeline" in opened.panels()
    assert opened.raised == []


def test_a_corpus_that_has_never_run_offers_to_run_it(page) -> None:
    """The button this whole panel exists for."""
    opened = page(pipeline_api=client())

    run = opened.button("Run the pipeline")

    assert run is not None and run.disabled is False


def test_run_is_refused_while_a_run_is_already_going(page) -> None:
    """Two runs would take the same queues through at once."""
    opened = page(
        pipeline_api=client(
            orchestration=orchestration(
                running={"id": "run-7abcdef", "status": "STARTED"}
            )
        )
    )

    assert opened.button("Run the pipeline").disabled is True
    assert "run-7abc" in opened.text()


def test_a_deployment_with_no_orchestrator_says_why_rather_than_hiding_run(
    page,
) -> None:
    """`orchestration/` is optional, and its absence is explicable."""
    opened = page(
        pipeline_api=client(orchestration=orchestration(available=False, running=None))
    )

    run = opened.button("Run the pipeline")

    assert run is not None, "the control stays, so its help can say why"
    assert run.disabled is True
    assert "no orchestrator" in run.help


def test_the_queue_verbs_work_without_an_orchestrator(page) -> None:
    """They are queue verbs, and the queue is the backend's own."""
    opened = page(
        pipeline_api=client(orchestration=orchestration(available=False, running=None))
    )

    assert opened.button("Start every stage").disabled is False
    assert opened.button("Retry failures").disabled is False


def test_start_is_dead_when_nothing_is_ready(page) -> None:
    """A control that would move nothing is not offered."""
    opened = page(
        pipeline_api=client(
            stages=[{"stage": "parsing", "working": False, "rows": {"parsed": 3}}],
            failed=0,
        )
    )

    assert opened.button("Start every stage").disabled is True


def test_retry_is_dead_when_nothing_has_failed(page) -> None:
    """And live when something has."""
    opened = page(pipeline_api=client(failed=0))
    assert opened.button("Retry failures").disabled is True

    opened = page(pipeline_api=client(failed=3))
    assert opened.button("Retry failures").disabled is False
    assert "3" in opened.button("Retry failures").help


def test_stop_is_live_while_anything_is_moving(page) -> None:
    """Whether a run set it going or somebody pressed Start."""
    opened = page(pipeline_api=client(working=False))
    assert opened.button("Stop everything").disabled is True

    opened = page(pipeline_api=client(working=True))
    assert opened.button("Stop everything").disabled is False


# ── What the panel sends ──────────────────────────────────────────────────


def test_pressing_run_asks_for_a_run(page) -> None:
    """One call, and the page redraws under it."""
    scripted = client()

    page(pipeline_api=scripted).press("Run the pipeline")

    assert len(scripted.calls("run")) == 1


@pytest.mark.parametrize(
    ("label", "action"),
    [
        ("Start every stage", "start"),
        ("Stop everything", "stop"),
        ("Retry failures", "retry"),
    ],
)
def test_each_verb_sends_its_own_action(page, label: str, action: str) -> None:
    """The label and the action cannot drift apart."""
    scripted = client(working=True, failed=2)

    page(pipeline_api=scripted).press(label)

    assert scripted.calls("act")[0][1] == (action,)


def test_a_refusal_is_shown_rather_than_raised(page) -> None:
    """The backend's own wording, which says what to do about it."""
    refusing = Answers(
        state=pipeline(),
        run=(False, "run abc is already taking this corpus through."),
        act=(True, "done"),
        automate=(True, "Saved."),
    )

    opened = page(pipeline_api=refusing).press("Run the pipeline")

    assert opened.raised == []
    assert "already taking this corpus through" in opened.text()


def test_a_backend_that_is_not_answering_draws_a_caption_not_a_trace(page) -> None:
    """The panel polls, so this is the state it is most often in."""
    opened = page(pipeline_api=Answers(state={}))

    assert opened.raised == []
    assert "not answering" in opened.text()


# ── Unattended running ────────────────────────────────────────────────────


def test_the_two_triggers_are_folded_away_and_read_as_off(page) -> None:
    """Off is what ships, and turning one on is a decision made once."""
    opened = page(pipeline_api=client())

    folded = [one for one in opened.folds() if "without being asked" in one]

    assert folded == ["Run it without being asked"]


def test_a_trigger_that_is_on_is_named_on_the_fold(page) -> None:
    """So nobody has to open it to find out the pipeline is unattended."""
    opened = page(
        pipeline_api=client(
            orchestration=orchestration(
                automation={
                    "available": True,
                    "on_arrival": True,
                    "nightly": False,
                    "detail": None,
                }
            )
        )
    )

    assert any("on upload" in one for one in opened.folds())


def test_without_an_orchestrator_there_is_nothing_to_switch(page) -> None:
    """And the fold says that rather than offering a dead toggle."""
    opened = page(
        pipeline_api=client(orchestration=orchestration(available=False, running=None))
    )

    assert "No orchestrator" in opened.text()
