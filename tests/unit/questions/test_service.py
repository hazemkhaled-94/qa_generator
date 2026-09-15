"""What the service does with a topic, short of touching a database.

The paths worth covering here are the ones that produce nothing: a topic
somebody took out of coverage, a topic whose facts are all already asked
about, and a model that will not answer. Each has to leave the row in a state
something can move it out of, because a queue that never empties says nothing
and a row nothing can move is a stage that has stopped.
"""

from __future__ import annotations

import pytest
from factories import source

from database.qa_generator import Status
from llm.client import ModelUnavailable
from question_generation.config import Settings
from question_generation.models import CheckedQuestion, TopicToCover
from question_generation.service import QuestionGenerationService

SETTINGS = Settings(
    per_topic=4,
    sample_size=2,
    unanswerable_share=0.25,
    duplicate_cosine=0.93,
    embedding_model="stub",
    max_tokens=512,
    verifier_model="ollama/verifier",
)


class StubQueue:
    """A queue that answers from memory and records what it was told."""

    done = Status.GENERATED

    def __init__(self, topic: TopicToCover | None, facts=()) -> None:
        """Initialises the queue with one topic and its facts."""
        self._topic = topic
        self._facts = list(facts)
        self.stored: list[list[CheckedQuestion]] = []
        self.failed: list[tuple[int, str]] = []
        self.finished: list[int] = []

    def abandon(self) -> int:
        """Sweeps nothing; there is no earlier run here."""
        return 0

    def claim(self) -> TopicToCover | None:
        """Hands over the one topic, once."""
        claimed, self._topic = self._topic, None
        return claimed

    def facts(self, topic_id: int):
        """The facts this topic is the subject of."""
        return list(self._facts)

    def store(self, topic_id: int, questions) -> int:
        """Records what the topic produced and finishes it."""
        self.stored.append(list(questions))
        self.finished.append(topic_id)
        return len(questions)

    def fail(self, key: int, error: str) -> None:
        """Records the failure against the row."""
        self.failed.append((key, error))


class StubWriter:
    """A writer that answers, or refuses, and counts the asks."""

    def __init__(self, raises: Exception | None = None) -> None:
        """Initialises the writer with what it will do when asked."""
        self._raises = raises
        self.calls = 0

    @property
    def model(self) -> str:
        """The model this writer would call."""
        return "ollama/stub"

    def write(self, group, *, answerable: bool):
        """Writes one question, or refuses."""
        self.calls += 1
        if self._raises:
            raise self._raises
        from factories import candidate

        return candidate(
            question_text=f"Question {self.calls}?",
            target_answer=None if not answerable else "4 kg",
            answerable=answerable,
            facts=group,
        )


class StubChecker:
    """A checker that accepts everything and counts what it saw."""

    def __init__(self) -> None:
        """Initialises the checker with nothing checked yet."""
        self.seen_sizes: list[int] = []

    def check(self, candidate, seen=()) -> CheckedQuestion:
        """Accepts the candidate, noting how much this run had accepted."""
        self.seen_sizes.append(len(seen))
        return CheckedQuestion(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            difficulty=candidate.group.difficulty,
            language=candidate.group.language,
            status="accepted",
            rejected_reason=None,
            fact_ids=tuple(one.id for one in candidate.group.facts),
        )


def build(topic: TopicToCover | None, facts=(), writer=None, checker=None):
    """A service over a scripted queue and a scripted pair of collaborators."""
    queue = StubQueue(topic, facts)
    service = QuestionGenerationService(
        repository=queue,
        writer=writer or StubWriter(),
        checker=checker or StubChecker(),
        settings=SETTINGS,
    )
    return service, queue


def topic(**fields) -> TopicToCover:
    """One claimed topic, in coverage unless a test says otherwise."""
    return TopicToCover(
        id=fields.pop("id", 7),
        language=fields.pop("language", "en"),
        label=fields.pop("label", "Support"),
        include_in_coverage=fields.pop("include_in_coverage", True),
        **fields,
    )


def test_an_empty_queue_is_not_an_error(written=None) -> None:
    """A drain stops when the queue does."""
    service, queue = build(None)

    assert service.process_next() is None
    assert queue.stored == []


def test_a_topic_out_of_coverage_is_finished_with_nothing_written() -> None:
    """Somebody said this is not a subject, which is an answer and not work.

    Left `new` it would read as outstanding forever, and the model would be
    called for a topic a person deliberately excluded.
    """
    writer = StubWriter()
    service, queue = build(topic(include_in_coverage=False), [source(1)], writer=writer)

    service.process_next()

    assert queue.stored == [[]]
    assert queue.finished == [7]
    assert writer.calls == 0, "the model was called for an excluded topic"
    assert queue.failed == []


def test_a_topic_with_nothing_left_to_ask_about_is_finished_not_failed() -> None:
    """Every fact already carries an accepted question, which is success."""
    writer = StubWriter()
    service, queue = build(topic(), [], writer=writer)

    service.process_next()

    assert queue.stored == [[]]
    assert writer.calls == 0
    assert queue.failed == []


def test_a_topic_writes_no_more_than_it_was_asked_for() -> None:
    """QUESTIONS_PER_TOPIC times the topic count is what a run costs."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    service, queue = build(topic(), facts)

    service.process_next()

    assert len(queue.stored[0]) == SETTINGS.per_topic


def test_the_run_sees_what_it_has_already_accepted() -> None:
    """Nothing is stored until the topic is finished.

    Without carrying the accepted ones forward, the dedup gate compares each
    candidate only against what the database held when the run started.
    """
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    checker = StubChecker()
    service, _ = build(topic(), facts, checker=checker)

    service.process_next()

    assert checker.seen_sizes == [0, 1, 2, 3], checker.seen_sizes


def test_the_unanswerable_share_reaches_the_writer() -> None:
    """A quarter of four is one, and it is the fourth."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    service, queue = build(topic(), facts)

    service.process_next()

    written = queue.stored[0]
    assert [one.answerable for one in written] == [True, True, True, False]
    assert written[-1].target_answer is None


def test_a_model_that_will_not_answer_fails_the_topic(caplog) -> None:
    """Recorded against the row with a reason, so retry can move it."""
    writer = StubWriter(raises=ModelUnavailable("Timeout: the model did not answer"))
    service, queue = build(topic(), [source(1)], writer=writer)

    service.process_next()

    assert queue.stored == [], "a failed topic must not be marked done"
    assert len(queue.failed) == 1
    key, error = queue.failed[0]
    assert key == 7
    assert "did not answer" in error


def test_a_failure_names_the_kind_of_error_it_was() -> None:
    """One line on the row; the traceback goes to the log."""
    service, queue = build(
        topic(), [source(1)], writer=StubWriter(raises=RuntimeError("boom"))
    )

    service.process_next()

    assert queue.failed[0][1].startswith("RuntimeError:")


def test_a_drain_works_the_queue_until_it_is_empty() -> None:
    """Which is one topic here, and then nothing."""
    service, queue = build(topic(), [source(1)])

    assert service.drain() == 1
    assert queue.finished == [7]


@pytest.mark.parametrize("size", [0, -1])
def test_a_group_size_below_one_never_yields_an_empty_group(size) -> None:
    """An empty group has no language to write in and no spread to read.

    QUESTIONS_GROUP_SIZE is a number out of a settings file, and the one
    thing selection must not do with a silly one is hand back a group
    nothing downstream can use.
    """
    from question_generation.selection import samples

    formed = samples([source(n) for n in range(3)], wanted=5, size=size)

    assert formed, "every fact was dropped"
    assert all(one.facts for one in formed)
    assert all(one.language == "en" for one in formed)


def test_a_topics_lease_covers_every_call_it_will_make() -> None:
    """A lease sized for one model call fails a live worker mid-topic.

    Extraction reads one passage with one call and derives its lease from
    that. A topic is `per_topic` candidates, each a writer call and a
    verifier call, so the same arithmetic would sweep a worker that had
    barely started.
    """
    # One call's worst case: the timeout, on every attempt.
    call = 900.0 * 3

    held = SETTINGS.lease(call).total_seconds()

    assert held >= call * SETTINGS.per_topic * 2, "shorter than the work"


def test_the_lease_is_the_worst_case_and_not_a_multiple_of_it() -> None:
    """Padding a worst case is what makes a stranded row stay stranded.

    Nothing but the lease running out returns an `in_progress` row to the
    queue: stop moves `pending` and retry moves `failed`. So every hour
    added here is an hour a killed worker's topic cannot be picked up.
    """
    call = 900.0 * 3

    held = SETTINGS.lease(call).total_seconds()

    assert held == call * SETTINGS.per_topic * 2


def test_the_phrasing_gate_is_off_when_no_second_model_is_named() -> None:
    """The factory decides it, from the one setting that says so."""
    from dataclasses import replace

    from llm.config import Settings as ModelSettings
    from question_generation.verification import QuestionChecker

    model = ModelSettings(
        model="ollama/writer",
        base_url=None,
        structured_mode="JSON_SCHEMA",
        temperature=0.0,
        timeout_seconds=900.0,
        max_attempts=3,
    )

    def built(verifier: str | None) -> bool:
        """Whether a checker wired for this verifier may judge phrasing."""
        checker = QuestionChecker(
            embedder=object(),
            verifier=object(),
            nearest=lambda embedding: None,
            threshold=0.93,
            judge_phrasing=bool(
                replace(SETTINGS, verifier_model=verifier).verifier_model
            ),
        )
        return checker._judge_phrasing

    assert built("ollama/verifier") is True
    assert built(None) is False
    assert model.model == "ollama/writer"
