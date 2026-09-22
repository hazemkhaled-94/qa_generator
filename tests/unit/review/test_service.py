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
    id: str | None = None


class Records:
    """A dataset's records: read by calling, written by logging."""

    def __init__(self, held: list[Record]) -> None:
        """Initialises with what the dataset already holds."""
        self._held = held
        self.logged: list = []

    def __call__(self, with_responses: bool = False):
        """Every record in it."""
        del with_responses
        return iter(self._held)

    def log(self, records) -> None:
        """Records what a push wrote."""
        self.logged += list(records)


class Dataset:
    """A dataset holding some records."""

    def __init__(self, records: list[Record]) -> None:
        """Initialises with the records a pull will read."""
        self.records = Records(records)


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


# ── What a second push offers ──────────────────────────────────────────────


def asking(monkeypatch, held: list[Record], rows: list) -> tuple[Catalogs, Dataset]:
    """Points a push at a dataset already holding these records."""
    dataset = Dataset(held)
    monkeypatch.setattr("review.service.connect", lambda settings: Client(dataset))

    class Sampled:
        """A repository handing back a fixed sample."""

        def questions(self, sample: int, ids=None):
            """The rows, whatever was asked for."""
            del sample, ids
            return rows

    return Catalogs(facts=Sampled(), questions=Questions(), topics=Topics()), dataset


def row(question_id: int):
    """One question as the review reads it."""
    from review.records import QuestionRow

    return QuestionRow(
        id=question_id,
        question_text="What does the device weigh?",
        target_answer="4 kg",
        answerable=True,
        difficulty="easy",
        question_type="factoid",
        language="en",
        status="accepted",
        rejected_reason=None,
        facts=["The device weighs 4 kg."],
    )


def test_a_second_push_does_not_offer_a_question_already_answered(
    monkeypatch, settings
) -> None:
    """A sample that keeps drawing judged rows never reaches the rest.

    `facts` skips one through `reviewed_verdict`; a question carries no
    such column, so the answers Argilla already holds are what says so.
    """
    from review.service import push

    catalogs, dataset = asking(
        monkeypatch,
        [Record({"question_id": 7}, [submitted("verdict", "accepted")], id="7")],
        [row(7), row(8)],
    )

    assert push(datasets.QUESTIONS, settings, catalogs) == 1
    assert [one.id for one in dataset.records.logged] == ["8"]


def test_a_push_still_offers_a_question_somebody_only_started_on(
    monkeypatch, settings
) -> None:
    """A draft is not an answer, so the record is offered again."""
    from review.service import push

    catalogs, _ = asking(
        monkeypatch,
        [
            Record(
                {"question_id": 7}, [Response("verdict", "accepted", "draft")], id="7"
            )
        ],
        [row(7)],
    )

    assert push(datasets.QUESTIONS, settings, catalogs) == 1


def test_a_named_queue_is_pushed_whatever_has_been_answered(
    monkeypatch, settings
) -> None:
    """Naming an id asks for that row, not for a sample of what is left."""
    from review.service import push

    catalogs, dataset = asking(
        monkeypatch,
        [Record({"question_id": 7}, [submitted("verdict", "accepted")], id="7")],
        [row(7)],
    )

    assert push(datasets.QUESTIONS, settings, catalogs, ids=[7]) == 1
    assert [one.id for one in dataset.records.logged] == ["7"]


def test_the_response_status_is_read_either_way_it_is_spelled() -> None:
    """A response status is read either way the SDK spells it.

    It gives a str on one path and an enum on another, and a comparison
    that only handled one silently wrote nothing.
    """
    record = Record({}, [Response("verdict", "accepted", "ResponseStatus.submitted")])

    assert _answer(record, "verdict") == "accepted"
