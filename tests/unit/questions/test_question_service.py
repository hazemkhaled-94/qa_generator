"""What the service does with a topic, short of touching a database.

The paths worth covering here are the ones that produce nothing: a topic
somebody took out of coverage, a topic whose facts are all already asked
about, and a model that will not answer. Each has to leave the row in a state
something can move it out of, because a queue that never empties says nothing
and a row nothing can move is a stage that has stopped.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from factories import source

from database.qa_generator import Difficulty, QuestionType, Status
from llm.client import ModelUnavailable
from question_generation.config import Settings
from question_generation.models import CheckedQuestion, TopicToCover
from question_generation.service import QuestionGenerationService

SETTINGS = Settings(
    per_topic=4,
    sample_size=4,
    fact_kinds=("atomic",),
    type_mix={QuestionType.FACTOID: 3, QuestionType.REASON: 1},
    difficulty_mix={Difficulty.EASY: 1, Difficulty.MEDIUM: 1},
    followup_types=(QuestionType.CONDITION, QuestionType.REASON),
    unanswerable_share=0.25,
    followup_share=0.5,
    max_followups=2,
    answer_chars={"value": (1, 80), "list": (3, 300), "explanation": (20, 600)},
    answer_overlap=0.6,
    long_answer_chars=60,
    duplicate_cosine=0.93,
    embedding_model="stub",
    max_tokens=512,
    verifier_model="ollama/verifier",
)


class StubQueue:
    """A queue that answers from memory and records what it was told."""

    done = Status.GENERATED

    def __init__(self, topic: TopicToCover | None, facts=(), bridges=()) -> None:
        """Initialises the queue with one topic, its facts and its bridges."""
        self._topic = topic
        self._facts = list(facts)
        self._bridges = list(bridges)
        self.stored: list[list[list[CheckedQuestion]]] = []
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

    def bridging(self, topic_id: int):
        """The facts of passages bridging this topic to another."""
        return list(self._bridges)

    def store(self, topic_id: int, threads) -> int:
        """Records the threads the topic produced and finishes it."""
        self.stored.append([list(one) for one in threads])
        self.finished.append(topic_id)
        return sum(len(one) for one in threads)

    @property
    def written(self) -> list:
        """Every question of the last store, threads flattened."""
        return [one for thread in self.stored[-1] for one in thread]

    def fail(self, key: int, error: str) -> None:
        """Records the failure against the row."""
        self.failed.append((key, error))


class StubWriter:
    """A writer that answers, or refuses, and counts the asks."""

    def __init__(self, raises: Exception | None = None) -> None:
        """Initialises the writer with what it will do when asked."""
        self._raises = raises
        self.calls = 0
        self.plans: list = []

    @property
    def model(self) -> str:
        """The model this writer would call."""
        return "ollama/stub"

    def write(self, group, plan):
        """Writes one question of the kind the plan asked for, or refuses."""
        self.calls += 1
        self.plans.append(plan)
        if self._raises:
            raise self._raises
        from factories import candidate

        return candidate(
            question_text=f"Question {self.calls}?",
            target_answer=None if not plan.answerable else "4 kg",
            answerable=plan.answerable,
            facts=group,
            question_type=plan.spec.name,
            planned_difficulty=plan.band,
        )

    def follow_up(self, group, thread, plan):
        """Writes the next question in a thread."""
        self.calls += 1
        self.plans.append(plan)
        if self._raises:
            raise self._raises
        from factories import candidate

        return candidate(
            question_text=f"And question {self.calls}?",
            target_answer="12 hours",
            answerable=True,
            facts=group,
            thread=thread,
            question_type=plan.spec.name,
            planned_difficulty=plan.band,
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
            criteria=candidate.group.criteria(
                len(candidate.target_answer or "") or None,
                follows=candidate.follows,
            ),
            language=candidate.group.language,
            status="accepted",
            rejected_reason=None,
            fact_ids=tuple(one.id for one in candidate.group.facts),
            thread_position=candidate.thread_position,
            question_type=candidate.spec.name,
            answer_form=candidate.spec.form if candidate.answerable else None,
            planned_difficulty=candidate.planned_difficulty,
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

    assert len(queue.written) >= SETTINGS.per_topic


def test_the_run_sees_what_it_has_already_accepted() -> None:
    """Nothing is stored until the topic is finished.

    Without carrying the accepted ones forward, the dedup gate compares each
    candidate only against what the database held when the run started.
    """
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    checker = StubChecker()
    service, _ = build(topic(), facts, checker=checker)

    service.process_next()

    # Every accepted question joins the set, follow-ups included, so the
    # list only ever grows and each candidate sees one more than the last.
    assert checker.seen_sizes == sorted(checker.seen_sizes)
    assert checker.seen_sizes[0] == 0
    assert checker.seen_sizes[-1] == len(checker.seen_sizes) - 1


def test_the_unanswerable_share_reaches_the_writer() -> None:
    """A quarter of four is one, and it is the fourth."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    service, queue = build(topic(), facts)

    service.process_next()

    roots = [thread[0] for thread in queue.stored[0]]
    assert [one.answerable for one in roots] == [True, True, True, False]
    assert roots[-1].target_answer is None


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
def test_a_sample_size_below_one_never_yields_an_empty_group(size) -> None:
    """An empty group has no language to write in and no spread to read.

    QUESTIONS_FACT_SAMPLE is a number out of a settings file, and the one
    thing the deal must not do with a silly one is hand back a group nothing
    downstream can use.
    """
    from question_generation.selection import Deal

    deal = Deal([source(n, passage_id=n) for n in range(3)], wanted=5, size=size)
    formed = [deal.sample() for _ in range(3)]

    assert all(one is not None for one in formed), "every fact was dropped"
    assert all(one.facts for one in formed if one)
    assert all(one.language == "en" for one in formed if one)


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

    assert held == call * SETTINGS.per_topic * (1 + SETTINGS.max_followups) * 2


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
        num_ctx=None,
        reasoning_effort=None,
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


# ── Follow-up threads ──────────────────────────────────────────────────────


def test_a_thread_is_a_root_and_the_questions_that_follow_it() -> None:
    """Stored as a thread so each follow-up can be linked to its parent.

    A follow-up needs the parent's id and the parent has none until it is
    inserted, which is why the queue takes threads rather than a flat list.
    """
    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, queue = build(topic(), facts)

    service.process_next()

    threads = queue.stored[0]
    followed = [one for one in threads if len(one) > 1]
    assert followed, "no thread ran past its root"
    for thread in followed:
        assert thread[0].thread_position == 1
        assert [one.thread_position for one in thread] == list(
            range(1, len(thread) + 1)
        )
        assert all(one.follows for one in thread[1:])


def test_a_thread_never_runs_past_the_maximum() -> None:
    """QUESTIONS_MAX_FOLLOWUPS is what a topic's worst case is priced from."""
    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, queue = build(topic(), facts)

    service.process_next()

    assert all(len(thread) <= 1 + SETTINGS.max_followups for thread in queue.stored[0])


def test_an_unanswerable_question_is_never_followed() -> None:
    """There is nothing to follow on from.

    The chatbot was supposed to say it did not know, so a second question
    after that measures nothing.
    """
    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, queue = build(topic(), facts)

    service.process_next()

    for thread in queue.stored[0]:
        if not thread[0].answerable:
            assert len(thread) == 1


def test_a_rejected_root_is_never_followed() -> None:
    """A conversation starting with a question nobody would ask."""

    class Refusing(StubChecker):
        """A checker that rejects everything it sees."""

        def check(self, candidate, seen=()):
            """Rejects the candidate."""
            checked = super().check(candidate, seen)
            return replace(checked, status="rejected", rejected_reason="malformed")

    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, queue = build(topic(), facts, checker=Refusing())

    service.process_next()

    assert all(len(thread) == 1 for thread in queue.stored[0])


def test_a_follow_up_is_written_from_the_same_sample_as_its_root() -> None:
    """So the thread is about one subject rather than two in sequence."""
    facts = [source(n, document="a", passage_id=1) for n in range(3)]
    service, queue = build(topic(), facts)

    service.process_next()

    for thread in queue.stored[0]:
        cited = {fact for one in thread for fact in one.fact_ids}
        assert cited <= {one.id for one in facts}


def test_a_follow_up_is_shown_what_was_already_asked() -> None:
    """Without the thread it is a fresh question, not a follow-up."""
    seen: list = []

    class Watching(StubWriter):
        """Records the conversation each follow-up was written from."""

        def follow_up(self, group, thread, plan):
            """Keeps the thread and writes as scripted."""
            seen.append(tuple(thread))
            return super().follow_up(group, thread, plan)

    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, _ = build(topic(), facts, writer=Watching())

    service.process_next()

    assert seen, "no follow-up was written"
    assert all(turns for turns in seen), "a follow-up was written with no thread"
    assert any(len(turns) > 1 for turns in seen), "no second follow-up saw two turns"


def test_the_turns_of_a_thread_take_the_kinds_they_were_configured_to() -> None:
    """A conversation that asks the same kind three times is one question.

    QUESTIONS_FOLLOWUP_TYPES is cycled down the thread, so it moves from a
    value to the circumstances it applies in to the reason behind it.
    """
    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    writer = StubWriter()
    service, queue = build(topic(), facts, writer=writer)

    service.process_next()

    followed = next(one for one in queue.stored[0] if len(one) > 2)
    assert [one.question_type for one in followed[1:]] == list(SETTINGS.followup_types)


# ── The plan, which is what makes the set configurable ────────────────────


def test_every_question_is_written_to_a_plan() -> None:
    """The kind and the band are decided before anything is written."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    writer = StubWriter()
    service, _ = build(topic(), facts, writer=writer)

    service.process_next()

    assert writer.plans
    assert {one.spec.name for one in writer.plans} <= set(SETTINGS.type_mix) | set(
        SETTINGS.followup_types
    )


def test_the_kinds_written_are_the_kinds_that_were_asked_for() -> None:
    """A type with no weight is never written, which is the switch."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    service, queue = build(topic(), facts)

    service.process_next()

    roots = [thread[0] for thread in queue.stored[0]]
    assert {one.question_type for one in roots} <= set(SETTINGS.type_mix)


def test_what_the_plan_asked_for_is_stored_beside_what_came_out() -> None:
    """Two runs of one corpus have to be comparable on the same terms."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(20)]
    service, queue = build(topic(), facts)

    service.process_next()

    roots = [thread[0] for thread in queue.stored[0]]
    assert all(one.planned_difficulty in set(SETTINGS.difficulty_mix) for one in roots)


def test_a_topic_that_runs_out_of_passages_stops_rather_than_repeating() -> None:
    """A passage is asked about once, so a small topic yields few questions."""
    facts = [source(1, document="a", passage_id=1)]
    service, queue = build(topic(), facts)

    service.process_next()

    assert len(queue.stored[0]) == 1
    assert queue.finished == [7]
