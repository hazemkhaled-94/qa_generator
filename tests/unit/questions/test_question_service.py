"""What the service does with a topic, short of touching a database.

The paths worth covering here are the ones that produce nothing: a topic
somebody took out of coverage, a topic whose facts are all already asked
about, and a model that will not answer. Each has to leave the row in a state
something can move it out of, because a queue that never empties says nothing
and a row nothing can move is a stage that has stopped.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from factories import source

from database.qa_generator import (
    Difficulty,
    QuestionRejection,
    QuestionType,
    Status,
)
from llm.client import ModelUnavailable
from question_generation.config import Settings
from question_generation.models import CheckedQuestion, TopicToCover
from question_generation.queue import QuestionQueue
from question_generation.service import QuestionGenerationService
from stages.queue import StageQueue
from stages.service import HEARTBEAT_SECONDS

SETTINGS = Settings(
    per_topic=4,
    sample_size=4,
    samples_per_passage=1,
    fact_kinds=("atomic",),
    type_mix={QuestionType.FACTOID: 3, QuestionType.REASON: 1},
    difficulty_mix={Difficulty.EASY: 1, Difficulty.MEDIUM: 1},
    followup_types=(QuestionType.CONDITION, QuestionType.REASON),
    unanswerable_share=0.25,
    followup_share=0.5,
    max_followups=2,
    # Nothing is asked twice here. A retry is another model call, and these
    # tests count the calls a topic costs.
    retries=0,
    answer_chars={"value": (1, 80), "list": (3, 300), "explanation": (20, 600)},
    explanation_chars=(150, 900),
    answer_coverage=0.0,
    party_density=0.0,
    meets_floor=0.0,
    answer_overlap=0.6,
    off_topic_overlap=0.3,
    elsewhere_passages=0,
    long_answer_chars=60,
    boilerplate_cosine=0.0,
    duplicate_cosine=0.93,
    release_size=0,
    release_unanswerable=0.1,
    release_difficulty={Difficulty.EASY: 1, Difficulty.MEDIUM: 1, Difficulty.HARD: 1},
    embedding_model="stub",
    max_tokens=512,
    model=None,
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

    def touch(self) -> bool:
        """Refreshes the claim this queue holds."""
        return True

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

    def follow_up(self, group, thread, plan, root_facts=(), parent_passages=()):
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


def build(
    topic: TopicToCover | None, facts=(), writer=None, checker=None, settings=None
):
    """A service over a scripted queue and a scripted pair of collaborators."""
    queue = StubQueue(topic, facts)
    service = QuestionGenerationService(
        repository=queue,
        writer=writer or StubWriter(),
        checker=checker or StubChecker(),
        settings=settings or SETTINGS,
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


def test_a_topic_is_held_by_the_beat_and_not_by_the_lease() -> None:
    """What a stranded topic used to cost, and no longer does.

    Nothing but the lease running out used to return an `in_progress` row:
    stop moves `pending`, retry moves `failed`, rerun skips what is held.
    A lease sized for the worst case a topic could cost was therefore also
    how long a killed worker's topic stayed unreachable - days of it.

    The queue takes the default now, and a worker says it is still there
    every `HEARTBEAT_SECONDS`, so the work may run as long as it likes.
    """
    assert QuestionQueue().lease == StageQueue.lease
    assert StageQueue.lease <= timedelta(minutes=5)
    assert HEARTBEAT_SECONDS * 2 < StageQueue.lease.total_seconds()


def test_the_phrasing_gate_is_off_when_one_model_does_both() -> None:
    """The factory decides it, from the two models it resolved.

    Through `models`, which is what the factory calls, rather than through a
    copy of the rule here: the rule changed once already, when the writer
    became overridable, and a test carrying its own copy went on passing.

    Every combination of the two settings is in
    tests/unit/questions/test_models.py. This is the wiring: that the flag
    the checker ends up holding is the one the resolution produced.
    """
    from llm.config import Settings as ModelSettings
    from question_generation.checker import QuestionChecker
    from question_generation.factory import models

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
        writer, checker_model = models(
            replace(SETTINGS, model=None, verifier_model=verifier), model
        )
        checker = QuestionChecker(
            embedder=object(),
            verifier=object(),
            nearest=lambda embedding: None,
            threshold=0.93,
            judge_phrasing=writer.model != checker_model.model,
        )
        return checker._judge_phrasing

    assert built("ollama/verifier") is True
    assert built(None) is False
    # Naming the writer's own model is the same as naming none, which the
    # setting being set would have hidden.
    assert built("ollama/writer") is False
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

        def follow_up(self, group, thread, plan, root_facts=(), parent_passages=()):
            """Keeps the thread and writes as scripted."""
            seen.append(tuple(thread))
            return super().follow_up(group, thread, plan, root_facts, parent_passages)

    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    service, _ = build(topic(), facts, writer=Watching())

    service.process_next()

    assert seen, "no follow-up was written"
    assert all(turns for turns in seen), "a follow-up was written with no thread"
    assert any(len(turns) > 1 for turns in seen), "no second follow-up saw two turns"


def test_the_turns_of_a_thread_take_the_kinds_they_were_configured_to() -> None:
    """A conversation that asks the same kind three times is one question.

    QUESTIONS_FOLLOWUP_TYPES is cycled down the thread, so it moves from a
    value to the circumstances it applies in to the reason behind it. The
    cycle starts where the thread sits among the followed ones rather than
    always at the first type, so a list longer than QUESTIONS_MAX_FOLLOWUPS
    is a rotation and not a prefix with a dead tail.
    """
    facts = [source(n, document="a", passage_id=n) for n in range(8)]
    writer = StubWriter()
    service, queue = build(topic(), facts, writer=writer)

    service.process_next()

    names = list(SETTINGS.followup_types)
    threads = [one for one in queue.stored[0] if len(one) > 1]
    assert threads, "no thread was followed up"
    for thread in threads:
        turns = [one.question_type for one in thread[1:]]
        start = names.index(turns[0])
        assert turns == [names[(start + n) % len(names)] for n in range(len(turns))], (
            "a thread repeated a kind instead of cycling the configured list"
        )


def test_every_configured_follow_up_kind_is_reached_over_a_run() -> None:
    """The defect the rotation fixes.

    A thread runs QUESTIONS_MAX_FOLLOWUPS turns. Starting every one of them
    at the first configured type means nothing past that many is ever
    written, however many are configured: three types and two turns left the
    third unwritten in a whole corpus.
    """
    names = (QuestionType.CONDITION, QuestionType.REASON, QuestionType.COMPARISON)
    # Enough slots that three threads are followed: the share takes every
    # other slot and the unanswerable ones are never followed, so a topic of
    # four reaches one thread and rotates nothing.
    settings = replace(SETTINGS, followup_types=names, max_followups=2, per_topic=12)
    facts = [source(n, document="a", passage_id=n) for n in range(40)]
    service, queue = build(topic(), facts, writer=StubWriter(), settings=settings)

    service.process_next()

    written = {
        one.question_type for thread in queue.stored[0] for one in thread if one.follows
    }
    assert set(names) <= written, f"never wrote {set(names) - written}"


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


# ── Asking again ───────────────────────────────────────────────────────────


def test_a_gate_the_writer_could_have_satisfied_is_worth_asking_again() -> None:
    """43 of 155 rejections over one corpus were of that kind.

    A question thrown away for how it was written rather than for what it
    was about, where saying what went wrong costs one call.
    """
    from question_generation.service import again

    refused = replace(
        accepted_question(),
        status="rejected",
        rejected_reason=QuestionRejection.COMPOUND,
    )

    assert again(refused, easy_plan()) is not None


def test_a_duplicate_is_not_worth_asking_again() -> None:
    """The same sample asked twice gives another twin of the same question."""
    from question_generation.service import again

    refused = replace(
        accepted_question(),
        status="rejected",
        rejected_reason=QuestionRejection.DUPLICATE,
    )

    assert again(refused, easy_plan()) is None


def test_a_question_that_stood_is_left_alone() -> None:
    """Nothing to fix, and a second call would only risk the first draft."""
    from question_generation.service import again

    assert again(accepted_question(), easy_plan()) is None


def test_a_question_narrower_than_its_slot_is_asked_to_widen() -> None:
    """The other half of the difficulty problem.

    The writer is offered two passages, takes the escape hatch its own rules
    give it, cites one fact, and the question bands easy. Planned medium
    collapsed to easy 83 times over one corpus.
    """
    from question_generation.service import again

    assert again(accepted_question(), wide_plan()) is not None


def test_a_question_that_met_its_band_is_not_asked_to_widen() -> None:
    """The band is a request, and this one was granted."""
    from question_generation.service import again

    assert again(accepted_question(band=Difficulty.MEDIUM), wide_plan()) is None


def test_a_single_passage_slot_is_never_asked_to_widen() -> None:
    """It was never offered a second passage to use."""
    from question_generation.service import again

    assert again(accepted_question(), easy_plan()) is None


def accepted_question(band: str = Difficulty.EASY) -> CheckedQuestion:
    """One accepted question, banded as a test asks."""
    from question_generation.models import Criteria

    return CheckedQuestion(
        question_text="How heavy is the device?",
        target_answer="4 kg",
        answerable=True,
        criteria=Criteria(
            passage_scope="single_passage",
            document_scope="single_document",
            topic_scope="single_topic",
            answer_chars=4,
            difficulty=band,
        ),
        language="en",
        status="accepted",
        rejected_reason=None,
        fact_ids=(1,),
        thread_position=1,
        question_type=QuestionType.FACTOID,
        answer_form="value",
        planned_difficulty=band,
    )


def easy_plan():
    """A slot asking for one passage, which cannot come out narrower."""
    from question_generation.planning import Plan
    from question_generation.selection import Shape
    from question_generation.types import SPECS

    return Plan(
        spec=SPECS[QuestionType.FACTOID],
        band=Difficulty.EASY,
        shape=Shape.SINGLE,
        answerable=True,
    )


def wide_plan():
    """A slot asking for two passages and the medium band."""
    from question_generation.planning import Plan
    from question_generation.selection import Shape
    from question_generation.types import SPECS

    return Plan(
        spec=SPECS[QuestionType.FACTOID],
        band=Difficulty.MEDIUM,
        shape=Shape.CROSS,
        answerable=True,
    )


def test_the_follow_up_share_is_a_share_of_the_ACCEPTED_roots() -> None:
    """Not of the plan's slots, which is what it used to be.

    Picked over the slots, the share was decided before the question was
    written and the thread only happened where that question was then
    accepted - so what landed was the share TIMES the acceptance rate. One
    run asked for 0.5 and got 448 threads from 1,209 accepted roots, which
    is 37%, and the number would have moved again if acceptance had.
    """

    class EverySecondRoot(StubChecker):
        """Rejects every other ROOT, so slot number and accepted number part.

        A follow-up carries the thread it joins, which is how this tells
        the two apart without counting its own follow-up checks into the
        alternation.
        """

        def __init__(self) -> None:
            super().__init__()
            self.roots = 0

        def check(self, candidate, seen=()):
            """Accepts, then rejects, then accepts - roots only."""
            checked = super().check(candidate, seen)
            if candidate.thread:
                return checked
            self.roots += 1
            if self.roots % 2 == 0:
                return replace(checked, status="rejected", rejected_reason="malformed")
            return checked

    facts = [source(n, document="a", passage_id=n) for n in range(24)]
    settings = replace(SETTINGS, followup_share=1.0, max_followups=1)
    service, queue = build(topic(), facts, checker=EverySecondRoot(), settings=settings)

    service.process_next()

    threads = queue.stored[0]
    accepted_roots = [one for one in threads if one[0].accepted]
    followed = [one for one in threads if len(one) > 1]

    assert accepted_roots, "the fixture accepted no root to follow"
    # A share of 1.0 means EVERY accepted root, whatever the rejections
    # between them did to the slot numbering.
    assert len(followed) == len(accepted_roots)


# ── The credits page, which is not a subject ───────────────────────────────


def test_a_passage_of_names_is_never_asked_about() -> None:
    """The furniture the repetition reading cannot see.

    Every document thanks DIFFERENT people, so an acknowledgements section
    is identical in shape and different in words and no cross-document
    similarity reaches it. `names_parties` reads what it is made of.
    """
    credits = "Graham Bath, Judy McKay, Tauhida Parveen, Rex Black, Mike Smith"
    facts = [
        source(1, passage_id=1, passage_text=credits, language="en"),
        source(2, passage_id=2, language="en"),
    ]
    settings = replace(SETTINGS, party_density=0.25)
    service, queue = build(topic(), facts, settings=settings)

    service.process_next()

    asked = {fact for written in queue.written for fact in written.fact_ids}
    assert 1 not in asked, "a question was written about the credits page"


def test_a_density_of_zero_asks_about_every_passage() -> None:
    """The default, so nothing is excluded unless a deployment says so."""
    credits = "Graham Bath, Judy McKay, Tauhida Parveen, Rex Black, Mike Smith"
    facts = [source(1, passage_id=1, passage_text=credits, language="en")]
    service, queue = build(
        topic(), facts, settings=replace(SETTINGS, party_density=0.0)
    )

    service.process_next()

    assert queue.written, "nothing was written at all"


class RetryingWriter(StubWriter):
    """A writer that takes a retry note and says which draft it wrote."""

    def write(self, group, plan, note: str = ""):
        """The base writer, with the note recorded and the text varied."""
        written = super().write(group, plan)
        self.notes.append(note)
        return replace(
            written, question_text=f"draft {len(self.notes)}: {written.question_text}"
        )

    def __init__(self) -> None:
        """Initialises the writer with nothing asked yet."""
        super().__init__()
        self.notes: list[str] = []


class RefusesTheFirstDraft(StubChecker):
    """Refuses the first draft under a gate a retry can answer, then accepts."""

    def check(self, candidate, seen=()) -> CheckedQuestion:
        """Rejects draft one as compound, which `again` has a note for."""
        checked = super().check(candidate, seen)
        if candidate.question_text.startswith("draft 1"):
            return replace(
                checked,
                status="rejected",
                rejected_reason=QuestionRejection.COMPOUND,
            )
        return checked


def test_a_retry_is_numbered_by_the_draft_that_wrote_it() -> None:
    """And keeps that number after the best draft is moved to the end.

    `_attempt` reorders its drafts so the kept one is written last, which
    is what makes a row's position in the list stop saying which attempt
    produced it. Without the column nothing does: both drafts are stored
    and neither says whether the retry was the one that worked.
    """
    service, queue = build(
        topic(),
        [source(1, passage_id=1, language="en")],
        writer=RetryingWriter(),
        checker=RefusesTheFirstDraft(),
        settings=replace(SETTINGS, retries=1, per_topic=1, followup_share=0.0),
    )

    service.process_next()

    drafts = [
        (one.attempt, one.status)
        for one in queue.written
        if one.question_text.startswith("draft ")
    ]
    assert (1, "rejected") in drafts, "the first draft was not recorded as attempt 1"
    assert (2, "accepted") in drafts, "the retry was not recorded as attempt 2"
    # The accepted one is stored last, so position and attempt disagree -
    # which is the whole reason the column exists.
    assert drafts[-1][0] == 2
