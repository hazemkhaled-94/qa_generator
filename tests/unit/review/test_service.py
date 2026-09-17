"""Bringing verdicts home.

What a pull must not do is most of this file. A draft is somebody
part-way through thinking, and writing one back records an opinion nobody
has finished having; an unanswered coverage question is not a vote to take
a topic out of coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("argilla", reason="the observability group is not installed")

from review import datasets
from review.service import Catalogs, _answer, pull


@dataclass
class Response:
    """One answer, submitted or still a draft."""

    question_name: str
    value: object
    status: str


@dataclass
class Record:
    """One Argilla record as the pull reads it."""

    metadata: dict
    responses: list[Response]


class Dataset:
    """A dataset holding some records."""

    def __init__(self, records: list[Record]) -> None:
        """Initialises with the records a pull will read."""
        self._records = records

    def records(self, with_responses: bool = False):
        """Every record in it."""
        del with_responses
        return iter(self._records)


class Client:
    """Answers `datasets(...)` with one dataset."""

    def __init__(self, dataset) -> None:
        """Initialises with the dataset to hand back."""
        self._dataset = dataset

    def datasets(self, name=None, workspace=None):
        """The dataset, whatever was asked for."""
        del name, workspace
        return self._dataset


class Facts:
    """Records what was written back."""

    def __init__(self) -> None:
        """Initialises with nothing written."""
        self.written: list[tuple[int, str]] = []

    def review_facts(self, verdicts) -> int:
        """Records the verdicts."""
        self.written += verdicts
        return len(verdicts)


class Questions:
    """Records the decisions made."""

    def __init__(self) -> None:
        """Initialises with nothing decided."""
        self.decided: list[tuple[int, str]] = []

    def decide(self, question_id: int, status: str):
        """Records one decision."""
        self.decided.append((question_id, status))
        return object()


class Topics:
    """Records the descriptions written."""

    def __init__(self) -> None:
        """Initialises with nothing described."""
        self.described: list[tuple[int, str | None, bool]] = []

    def describe(self, topic_id: int, *, label, include_in_coverage):
        """Records one description."""
        self.described.append((topic_id, label, include_in_coverage))
        return object()


@pytest.fixture
def settings(monkeypatch):
    """Settings that need no environment."""
    from review.config import Settings

    return Settings(
        api_url="http://argilla:6900",
        api_key="key",
        workspace="qa_generator",
        sample=10,
        reviewer=None,
    )


def wired(monkeypatch, records: list[Record]) -> Catalogs:
    """Points the service at a stub client holding these records."""
    monkeypatch.setattr(
        "review.service.connect", lambda settings: Client(Dataset(records))
    )
    return Catalogs(facts=Facts(), questions=Questions(), topics=Topics())


def submitted(question: str, value) -> Response:
    """One submitted answer."""
    return Response(question, value, "submitted")


def test_a_submitted_fact_verdict_is_written(monkeypatch, settings) -> None:
    """The whole point of the pull."""
    catalogs = wired(
        monkeypatch,
        [Record({"fact_id": 41}, [submitted("verdict", "rejected")])],
    )

    assert pull(datasets.FACTS, settings, catalogs) == 1
    assert catalogs.facts.written == [(41, "rejected")]


def test_a_draft_is_not_written(monkeypatch, settings) -> None:
    """Argilla saves one the moment a record is touched.

    Writing it back would record an opinion nobody has finished having,
    and the reviewer would find their half-thought already applied.
    """
    catalogs = wired(
        monkeypatch,
        [Record({"fact_id": 41}, [Response("verdict", "rejected", "draft")])],
    )

    assert pull(datasets.FACTS, settings, catalogs) == 0
    assert catalogs.facts.written == []


def test_a_record_nobody_answered_is_not_written(monkeypatch, settings) -> None:
    """Most of a pushed sample, most of the time."""
    catalogs = wired(monkeypatch, [Record({"fact_id": 41}, [])])

    assert pull(datasets.FACTS, settings, catalogs) == 0


def test_a_question_verdict_goes_through_the_existing_decision(
    monkeypatch, settings
) -> None:
    """A question verdict goes through the existing decision.

    The same write PATCH /questions/{id} makes, so a verdict given here
    and one given on the Questions page cannot differ.
    """
    catalogs = wired(
        monkeypatch,
        [Record({"question_id": 7}, [submitted("verdict", "accepted")])],
    )

    assert pull(datasets.QUESTIONS, settings, catalogs) == 1
    assert catalogs.questions.decided == [(7, "accepted")]


def test_a_corrected_label_is_written_with_its_coverage(monkeypatch, settings) -> None:
    """Both answers reach `describe`, which is what sets labelled_by."""
    catalogs = wired(
        monkeypatch,
        [
            Record(
                {"topic_id": 3},
                [
                    submitted("corrected_label", "Fees and charges"),
                    submitted("in_coverage", "no"),
                ],
            )
        ],
    )

    assert pull(datasets.TOPIC_LABELS, settings, catalogs) == 1
    assert catalogs.topics.described == [(3, "Fees and charges", False)]


def test_an_unanswered_coverage_question_leaves_the_topic_in_coverage(
    monkeypatch, settings
) -> None:
    """An unanswered coverage question leaves the topic in coverage.

    Only `no` takes one out. A reviewer who fixed the name and did not
    answer the second question has not voted to silence the topic.
    """
    catalogs = wired(
        monkeypatch,
        [Record({"topic_id": 3}, [submitted("corrected_label", "Fees")])],
    )

    pull(datasets.TOPIC_LABELS, settings, catalogs)

    assert catalogs.topics.described == [(3, "Fees", True)]


def test_a_missing_dataset_is_not_an_error(monkeypatch, settings) -> None:
    """Pulling before anything was pushed is a reasonable thing to do."""
    monkeypatch.setattr("review.service.connect", lambda settings: Client(None))
    catalogs = Catalogs(facts=Facts(), questions=Questions(), topics=Topics())

    assert pull(datasets.FACTS, settings, catalogs) == 0


def test_the_response_status_is_read_either_way_it_is_spelled() -> None:
    """A response status is read either way the SDK spells it.

    It gives a str on one path and an enum on another, and a comparison
    that only handled one silently wrote nothing.
    """
    record = Record({}, [Response("verdict", "accepted", "ResponseStatus.submitted")])

    assert _answer(record, "verdict") == "accepted"
