"""What a stage sends to Phoenix when it records its prompts.

`prompts.create` posts a NEW Phoenix version every time it is called and
compares nothing, so what decides whether this scales is which prompts are
handed to it. The table already knows: a start-up that changed nothing
wrote nothing, and nothing written is nothing to publish.

Get that wrong in the obvious direction - publish the catalogue rather than
what moved - and every run of every stage adds an identical version to
every prompt it sends, which is the growth `telemetry/projects.py` exists
to clean up in the other store.

Neither the database nor Phoenix is reached here.
"""

from __future__ import annotations

import pytest

from stages import prompts
from stages.prompts import Composed

ONE = Composed("atomic", "9", "ROLE\nA claims analyst.", "{{excerpt}}", {"a": 1})
TWO = Composed("digest", "3", "ROLE\nA summariser.", "{{excerpt}}", None)


@pytest.fixture
def published(monkeypatch) -> list[tuple[str, list[Composed]]]:
    """What reached the publisher, with the model settings stubbed out."""
    sent: list[tuple[str, list[Composed]]] = []

    def publish(service: str, composed, model) -> int:
        sent.append((service, list(composed)))
        return len(sent)

    monkeypatch.setattr("llm.config.Settings.load", staticmethod(lambda: None))
    monkeypatch.setattr("stages.publish.publish", publish)
    return sent


def test_only_the_prompts_that_moved_are_published(monkeypatch, published) -> None:
    """The one the write actually took, not the catalogue it was given."""
    monkeypatch.setattr(prompts, "_write", lambda service, held: [held[0]])

    assert prompts.record("extraction", [ONE, TWO]) == 1
    assert published == [("extraction", [ONE])]


def test_a_start_up_that_changed_nothing_publishes_nothing(
    monkeypatch, published
) -> None:
    """Which is every start-up after the first, and most of the runs."""
    monkeypatch.setattr(prompts, "_write", lambda service, held: [])

    assert prompts.record("extraction", [ONE, TWO]) == 0
    assert published == []


def test_a_publish_that_fails_does_not_fail_the_record(monkeypatch) -> None:
    """The prompt was sent and recorded; the copy beside the trace is not it."""

    def refuse(service, composed, model) -> int:
        raise ConnectionError("phoenix is down")

    monkeypatch.setattr("llm.config.Settings.load", staticmethod(lambda: None))
    monkeypatch.setattr(prompts, "_write", lambda service, held: list(held))
    monkeypatch.setattr("stages.publish.publish", refuse)

    assert prompts.record("extraction", [ONE, TWO]) == 2


def test_a_stage_with_nothing_to_record_reaches_neither_store(monkeypatch) -> None:
    """An empty catalogue is not a write and not a publish."""
    monkeypatch.setattr(
        prompts, "_write", lambda service, held: pytest.fail("wrote nothing")
    )

    assert prompts.record("extraction", []) == 0
