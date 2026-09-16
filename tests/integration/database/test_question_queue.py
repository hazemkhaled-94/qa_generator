"""Two queues on one table, against the database that has to keep them apart.

`topics` holds both the request to refit and the topics a fit produced. Topic
modelling queues over the first; question generation queues over the second.
Every one of these is the same failure seen from a different verb: a question
worker treating the asking as a subject, or counting it as one.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, fitted, membership, passage, question, topic
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.qa_generator import QuestionFact, Status
from question_generation.repository import QuestionCatalog, QuestionQueue
from topic_modelling.repository import TopicQueue

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(engine, database):
    """Writes topics, passages and facts, and hands back what was written."""

    def write(
        *,
        topics: int = 1,
        request: bool = False,
        facts_per_topic: int = 2,
        documents: int = 1,
        validated: bool = True,
    ) -> dict:
        with Session(engine) as session:
            shas = [digest(letter) for letter in "abcdefgh"[:documents]]
            session.add_all(document(sha) for sha in shas)
            if request:
                session.add(topic(status=Status.PENDING))
            written = [fitted(index) for index in range(topics)]
            session.add_all(written)
            session.flush()

            facts: dict[int, list[int]] = {}
            for position, held in enumerate(written):
                facts[held.id] = []
                for n in range(facts_per_topic):
                    sha = shas[n % len(shas)]
                    at = passage(
                        sha,
                        ordinal=position * 10 + n + 1,
                        text="The device weighs 4 kg.",
                        language="en",
                    )
                    session.add(at)
                    session.flush()
                    session.add(membership(at.id, held.id))
                    drawn = fact(at.id, statement=f"Claim {position}-{n}.")
                    drawn.validated = validated
                    if not validated:
                        drawn.rejection_code = "copied"
                    session.add(drawn)
                    session.flush()
                    facts[held.id].append(drawn.id)
            session.commit()
            return {"topics": [held.id for held in written], "facts": facts}

    return write


def statuses(engine) -> dict[str, int]:
    """Counts every row of topics by its question status, requests included."""
    with engine.connect() as connection:
        return dict(
            connection.execute(
                text("SELECT question_status, count(*) FROM topics GROUP BY 1")
            ).all()
        )


# ── The base predicate ─────────────────────────────────────────────────────


def test_start_never_queues_the_request_to_refit(corpus, engine) -> None:
    """The failure this whole predicate exists to prevent.

    Without it the fit request is queued as though it were a subject, and a
    question worker claims the row the topic worker is waiting for.
    """
    corpus(topics=2, request=True)

    assert QuestionQueue().start() == 2
    assert statuses(engine) == {Status.PENDING: 2, Status.NEW: 1}


def test_a_claim_never_takes_the_request_to_refit(corpus, engine) -> None:
    """Even when it is the only row in a claimable state."""
    corpus(topics=0, request=True)
    with engine.begin() as connection:
        connection.execute(text("UPDATE topics SET question_status = 'pending'"))

    assert QuestionQueue().claim() is None


def test_the_request_is_counted_by_neither_queue_as_the_other_one(
    corpus, engine
) -> None:
    """Each queue reports its own rows, and a page shows what it reports."""
    corpus(topics=3, request=True)

    assert sum(QuestionQueue().counts_by_status().values()) == 3
    # Topic modelling counts every row, because `modelled` is its topic count
    # and the rest describe the fit. That is its own contract, not this one's.
    assert sum(TopicQueue().counts_by_status().values()) == 4


def test_reset_and_stop_and_retry_all_leave_the_request_alone(corpus, engine) -> None:
    """Every inherited verb carries the predicate, not just the ones used."""
    corpus(topics=1, request=True)
    queue = QuestionQueue()

    queue.start()
    queue.stop()
    queue.retry()
    queue.reset()

    assert statuses(engine)[Status.NEW] == 1, "the request was moved"


def test_abandon_never_sweeps_the_request(corpus, engine) -> None:
    """A fit in progress is the topic worker's, and stays its business."""
    corpus(topics=0, request=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE topics SET question_status = 'in_progress', "
                "question_claimed_at = now() - interval '9 hours'"
            )
        )

    assert QuestionQueue().abandon() == 0


# ── The queue itself ───────────────────────────────────────────────────────


def test_a_claim_reads_the_topic_it_will_write_about(corpus) -> None:
    """The label and the coverage flag decide what the worker does next."""
    corpus(topics=1)
    QuestionQueue().start()

    claimed = QuestionQueue().claim()

    assert claimed is not None
    assert claimed.language == "en"
    assert claimed.include_in_coverage is True


def test_two_workers_never_take_the_same_topic(corpus) -> None:
    """FOR UPDATE SKIP LOCKED, over a queue of two dozen rows at most."""
    corpus(topics=3)
    queue = QuestionQueue()
    queue.start()

    taken = [queue.claim(), queue.claim(), queue.claim()]

    assert all(one is not None for one in taken)
    assert len({one.id for one in taken if one}) == 3


def test_a_claim_that_outlived_its_lease_is_failed(corpus, engine) -> None:
    """A topic costs many model calls, so its lease is long and still finite."""
    corpus(topics=1)
    QuestionQueue().start()
    QuestionQueue().claim()
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE topics SET question_claimed_at = now() - interval '30 days'")
        )

    assert QuestionQueue().abandon() == 1
    assert statuses(engine)[Status.FAILED] == 1


def test_narrowing_reaches_one_topic_and_no_other(corpus, engine) -> None:
    """A topic is the smallest subject there is, so it is the only scope."""
    written = corpus(topics=3)
    queue = QuestionQueue()

    moved = queue.start(queue.narrowed("topic", str(written["topics"][0])))

    assert moved == 1
    assert statuses(engine) == {Status.PENDING: 1, Status.NEW: 2}


# ── Which facts a topic is the subject of ──────────────────────────────────


def test_a_topic_reads_the_facts_of_the_passages_it_is_strongest_in(corpus) -> None:
    """A fact reaches its topic through its passage, never through a column."""
    written = corpus(topics=2, facts_per_topic=2)

    facts = QuestionQueue().facts(written["topics"][0])

    assert {one.id for one in facts} == set(written["facts"][written["topics"][0]])


def test_a_rejected_fact_is_never_written_about(corpus) -> None:
    """Only facts that passed every check are usable here."""
    written = corpus(topics=1, validated=False)

    assert QuestionQueue().facts(written["topics"][0]) == []


def test_a_passage_with_no_language_yields_no_facts(corpus, engine) -> None:
    """A question is written in one, and the column holding it is NOT NULL."""
    written = corpus(topics=1)
    with engine.begin() as connection:
        connection.execute(text("UPDATE passages SET language = NULL"))

    assert QuestionQueue().facts(written["topics"][0]) == []


def test_a_fact_an_accepted_question_already_rests_on_is_skipped(
    corpus, engine
) -> None:
    """What makes a second run cheap after the topics are fitted again.

    Without it the writer is paid for once per duplicate, and the dedup gate
    throws away work that was already bought.
    """
    written = corpus(topics=1, facts_per_topic=2)
    covered = written["facts"][written["topics"][0]][0]
    with Session(engine) as session:
        asked = question(status="accepted")
        session.add(asked)
        session.flush()
        session.execute(
            text("INSERT INTO question_facts (question_id, fact_id) VALUES (:q, :f)"),
            {"q": asked.id, "f": covered},
        )
        session.commit()

    left = QuestionQueue().facts(written["topics"][0])

    assert covered not in {one.id for one in left}
    assert len(left) == 1


def test_a_fact_only_a_rejected_question_rests_on_is_written_about_again(
    corpus, engine
) -> None:
    """A rejected question covers nothing; the fact is still unasked."""
    written = corpus(topics=1, facts_per_topic=1)
    uncovered = written["facts"][written["topics"][0]][0]
    with Session(engine) as session:
        asked = question(status="rejected", rejected_reason="duplicate")
        session.add(asked)
        session.flush()
        session.execute(
            text("INSERT INTO question_facts (question_id, fact_id) VALUES (:q, :f)"),
            {"q": asked.id, "f": uncovered},
        )
        session.commit()

    assert [one.id for one in QuestionQueue().facts(written["topics"][0])] == [
        uncovered
    ]


# ── Storing what a topic produced ──────────────────────────────────────────


def test_storing_writes_the_questions_their_links_and_finishes_the_topic(
    corpus, engine
) -> None:
    """One transaction: a topic marked done with no questions is a lie."""
    from question_generation.models import CheckedQuestion, criteria_of

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    facts = tuple(written["facts"][topic_id])
    QuestionQueue().start()
    QuestionQueue().claim()

    stored = QuestionQueue().store(
        topic_id,
        [
            [
                CheckedQuestion(
                    question_text="What does the device weigh?",
                    target_answer="4 kg",
                    answerable=True,
                    criteria=criteria_of(
                        passages=2, documents=1, topics=1, answer_chars=4
                    ),
                    language="en",
                    status="accepted",
                    rejected_reason=None,
                    fact_ids=facts,
                    embedding=[0.1] * 1024,
                )
            ]
        ],
    )

    assert stored == 1
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT count(*) FROM question_facts")).scalar()
            == 2
        )
        assert (
            connection.execute(
                text("SELECT question_status FROM topics WHERE id = :id"),
                {"id": topic_id},
            ).scalar()
            == Status.GENERATED
        )


def test_a_second_run_adds_to_the_questions_rather_than_replacing_them(
    corpus, engine
) -> None:
    """Questions are append-only; a rejected one is the drop-rate evidence."""
    from question_generation.models import CheckedQuestion, criteria_of

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    facts = written["facts"][topic_id]
    queue = QuestionQueue()

    for position, fact_id in enumerate(facts):
        queue.store(
            topic_id,
            [
                [
                    CheckedQuestion(
                        question_text=f"Question {position}?",
                        target_answer="4 kg",
                        answerable=True,
                        criteria=criteria_of(
                            passages=1, documents=1, topics=1, answer_chars=4
                        ),
                        language="en",
                        status="accepted",
                        rejected_reason=None,
                        fact_ids=(fact_id,),
                    )
                ]
            ],
        )

    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM questions")).scalar() == 2


def test_the_dedup_probe_finds_the_nearest_accepted_question(corpus, engine) -> None:
    """And ignores the rejected ones, which are not in the benchmark."""
    with Session(engine) as session:
        session.add_all(
            [
                question(
                    question_text="near",
                    embedding=[1.0] + [0.0] * 1023,
                    status="accepted",
                ),
                question(
                    question_text="nearer but rejected",
                    embedding=[1.0] + [0.0] * 1023,
                    status="rejected",
                    rejected_reason="duplicate",
                ),
                question(
                    question_text="far",
                    embedding=[0.0] * 1023 + [1.0],
                    status="accepted",
                ),
            ]
        )
        session.commit()

    found = QuestionCatalog().nearest([1.0] + [0.0] * 1023)

    assert found is not None
    assert found.question_text == "near"
    assert found.similarity == pytest.approx(1.0, abs=1e-5)


def test_the_probe_can_be_told_to_look_only_at_earlier_questions(
    corpus, engine
) -> None:
    """Without it a re-check rejects both halves of a duplicate pair.

    Each finds the other, both are rejected, and the benchmark is left with
    neither rather than with one.
    """
    with Session(engine) as session:
        first = question(
            question_text="first", embedding=[1.0] + [0.0] * 1023, status="accepted"
        )
        second = question(
            question_text="second", embedding=[1.0] + [0.0] * 1023, status="accepted"
        )
        session.add_all([first, second])
        session.commit()
        ids = (first.id, second.id)

    catalog = QuestionCatalog()

    assert catalog.nearest([1.0] + [0.0] * 1023, before=ids[0]) is None
    found = catalog.nearest([1.0] + [0.0] * 1023, before=ids[1])
    assert found is not None and found.question_text == "first"


# ── Edges the outer joins and the empty cases leave ────────────────────────


def test_a_question_with_no_facts_is_still_read_back(engine, database) -> None:
    """The trigger deletes one when its LAST link goes, not when it has none.

    A question written with no citation is never deleted by anything, so an
    inner join would hide a row that is really there and a count taken over
    it would disagree with the table.
    """
    with Session(engine) as session:
        session.add(question(question_text="Adrift?"))
        session.commit()

    total, rows = QuestionCatalog().page()

    assert total == 1
    assert rows[0].facts == 0
    assert rows[0].documents == []
    assert rows[0].topics == []


def test_a_question_whose_passage_has_no_topic_lists_no_topic(engine, database) -> None:
    """Not a list holding a null, which is what the aggregate returns."""
    with Session(engine) as session:
        session.add(document(digest("a")))
        session.flush()
        at = passage(digest("a"), ordinal=1, language="en")
        session.add(at)
        session.flush()
        drawn = fact(at.id)
        session.add(drawn)
        asked = question()
        session.add(asked)
        session.flush()
        session.add(QuestionFact(question_id=asked.id, fact_id=drawn.id))
        session.commit()

    _, rows = QuestionCatalog().page()

    assert rows[0].topics == []
    assert rows[0].documents == [digest("a")]


def test_a_topic_with_no_memberships_has_no_facts_to_ask_about(corpus) -> None:
    """A fit that placed no passage in a topic leaves it with nothing."""
    written = corpus(topics=1, facts_per_topic=0)

    assert QuestionQueue().facts(written["topics"][0]) == []


def test_finishing_a_topic_that_produced_nothing_still_marks_it_done(
    corpus, engine
) -> None:
    """A topic out of coverage yields no questions and is not still to do.

    Left `new` it would read as work outstanding forever, and a queue that
    never empties says nothing about whether the stage has run.
    """
    written = corpus(topics=1)
    topic_id = written["topics"][0]
    QuestionQueue().start()
    QuestionQueue().claim()

    assert QuestionQueue().store(topic_id, []) == 0
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT question_status FROM topics WHERE id = :id"),
                {"id": topic_id},
            ).scalar()
            == Status.GENERATED
        )


def test_the_probe_answers_nothing_when_no_question_is_accepted(
    engine, database
) -> None:
    """The first question of the first run has nothing to be near."""
    with Session(engine) as session:
        session.add(
            question(
                question_text="rejected",
                embedding=[1.0] + [0.0] * 1023,
                status="rejected",
                rejected_reason="duplicate",
            )
        )
        session.commit()

    assert QuestionCatalog().nearest([1.0] + [0.0] * 1023) is None


def test_the_probe_ignores_a_question_with_no_embedding(engine, database) -> None:
    """A malformed question is rejected before it is ever embedded."""
    with Session(engine) as session:
        session.add(question(question_text="never embedded", status="accepted"))
        session.commit()

    assert QuestionCatalog().nearest([1.0] + [0.0] * 1023) is None


def test_deciding_a_question_that_does_not_exist_answers_nothing(
    engine, database
) -> None:
    """Which is what the route turns into a 404 rather than a silent success."""
    assert QuestionCatalog().decide(999_999, "accepted") is None


def test_a_re_check_rejects_a_question_whose_evidence_moved(engine, database) -> None:
    """A cross-document question that lost a citation is not deleted.

    The trigger takes a question only when its last fact goes, so this one
    survives with a difficulty that stopped being true - which is quieter
    than a deletion and worse.
    """
    from question_generation.config import Settings
    from question_generation.service import reverify

    with Session(engine) as session:
        session.add_all([document(digest("a")), document(digest("b"))])
        session.flush()
        here = passage(digest("a"), ordinal=1, language="en")
        there = passage(digest("b"), ordinal=1, language="en")
        session.add_all([here, there])
        session.flush()
        facts = [fact(here.id), fact(there.id, statement="Reviewed yearly.")]
        session.add_all(facts)
        asked = question(
            question_text="What is it and how often is it reviewed?",
            target_answer="4 kg, yearly",
            status="accepted",
            difficulty="hard",
            passage_scope="multi_passage",
            document_scope="cross_document",
            topic_scope="single_topic",
            answer_chars=12,
        )
        session.add(asked)
        session.flush()
        session.add_all(QuestionFact(question_id=asked.id, fact_id=f.id) for f in facts)
        session.commit()
        asked_id, lost = asked.id, facts[1].id

    settings = Settings(
        per_topic=4,
        sample_size=4,
        fact_kinds=("atomic",),
        type_mix={"factoid": 1},
        difficulty_mix={"easy": 1},
        followup_types=("condition",),
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
    # Re-extracting one document deletes its facts; the other citation stays.
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM facts WHERE id = :id"), {"id": lost})

    assert reverify(QuestionCatalog(), settings) == 1
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT status, rejected_reason FROM questions WHERE id = :id"),
            {"id": asked_id},
        ).one()
    assert (row.status, row.rejected_reason) == ("rejected", "source_changed")


# ── The kind a question was asked to be ────────────────────────────────────


def test_a_question_keeps_the_kind_the_form_and_the_band_it_was_planned_as(
    corpus,
) -> None:
    """Three columns, and the CHECK constraints behind each.

    Without them a set cannot be filtered to its reasons, and a run cannot
    say whether the mix it was asked for is the mix it produced.
    """
    from question_generation.models import CheckedQuestion, criteria_of
    from question_generation.repository import QuestionCatalog

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    QuestionQueue().store(
        topic_id,
        [
            [
                CheckedQuestion(
                    question_text="Why is a request confirmed in writing?",
                    target_answer="so the agreed response time can be evidenced",
                    answerable=True,
                    criteria=criteria_of(
                        passages=1, documents=1, topics=1, answer_chars=44
                    ),
                    language="en",
                    status="accepted",
                    rejected_reason=None,
                    fact_ids=(written["facts"][topic_id][0],),
                    question_type="reason",
                    answer_form="explanation",
                    planned_difficulty="medium",
                )
            ]
        ],
    )

    total, rows = QuestionCatalog().page(question_type="reason")

    assert total == 1
    assert rows[0].question_type == "reason"
    assert rows[0].answer_form == "explanation"
    assert rows[0].planned_difficulty == "medium"
    assert QuestionCatalog().page(question_type="factoid")[0] == 0


def test_the_quality_report_counts_the_kinds_and_what_the_plan_asked_for(
    corpus,
) -> None:
    """The realised mix beside the planned one is the whole measurement."""
    from question_generation.models import CheckedQuestion, criteria_of
    from question_generation.repository import QuestionCatalog

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    facts = written["facts"][topic_id]
    QuestionQueue().store(
        topic_id,
        [
            [
                CheckedQuestion(
                    question_text=f"Question {position}?",
                    target_answer="4 kg",
                    answerable=True,
                    criteria=criteria_of(
                        passages=1, documents=1, topics=1, answer_chars=4
                    ),
                    language="en",
                    status="accepted",
                    rejected_reason=None,
                    fact_ids=(fact_id,),
                    question_type=kind,
                    answer_form="value",
                    # The first got the band it was planned as; the second
                    # was planned harder than it came out.
                    planned_difficulty="easy" if position == 0 else "hard",
                )
            ]
            for position, (fact_id, kind) in enumerate(
                zip(facts, ("factoid", "entity"), strict=False)
            )
        ],
    )

    quality = QuestionCatalog().quality()

    assert quality.question_type == {"factoid": 1, "entity": 1}
    assert quality.answer_form == {"value": 2}
    assert quality.planned_difficulty == {"easy": 1, "hard": 1}
    assert quality.planned_met == 1


# ── Which kinds of fact a question may be written from ────────────────────


def test_only_the_kinds_this_stage_can_use_are_offered(corpus, engine) -> None:
    """Extraction reads a passage four ways and one of them is a question seed.

    An outline is newline-separated `- ` bullets. Interpolated into the
    writer's numbered `[1] {fact}` list it spans several lines and breaks the
    numbering the writer is told to cite by; a summary is a paraphrase of the
    passage rather than a checked claim; a bridge rests on passages the
    verifier is never shown. Without this filter the first re-extraction that
    writes any of them silently changes what a question rests on.

    A bridge is askable and is covered separately below.
    """
    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE facts SET kind = 'outline' WHERE id = :id"),
            {"id": written["facts"][topic_id][0]},
        )

    offered = QuestionQueue().facts(topic_id)

    assert len(offered) == 1
    assert written["facts"][topic_id][0] not in {one.id for one in offered}


# ── Bridges, which rest on more than one passage ──────────────────────────


def _bridge(engine, *passage_ids: int, cited: bool = True) -> int:
    """Turns the first fact of these passages into a bridge resting on all.

    Args:
        engine: The engine the fixtures built.
        *passage_ids: The passages it rests on, anchor first.
        cited: Whether to record where in each it rests. False is a bridge
            drawn before the prompt said.

    Returns:
        The bridge fact's id.
    """
    anchor = passage_ids[0]
    with engine.begin() as connection:
        fact_id = connection.execute(
            text(
                "UPDATE facts SET kind = 'bridge' WHERE passage_id = :anchor "
                "RETURNING id"
            ),
            {"anchor": anchor},
        ).scalar_one()
        for position, passage_id in enumerate(passage_ids):
            connection.execute(
                text(
                    "INSERT INTO fact_passages (fact_id, passage_id, position, "
                    "sentence_ids, evidence_start, evidence_end) VALUES "
                    "(:fact, :passage, :position, :ids, :start, :end)"
                ),
                {
                    "fact": fact_id,
                    "passage": passage_id,
                    "position": position,
                    "ids": [0] if cited else None,
                    "start": 0 if cited else None,
                    "end": 23 if cited else None,
                },
            )
    return fact_id


def _passages_of(engine, topic_id: int) -> dict[int, list[int]]:
    """Which passages each offered fact rests on, by fact id."""
    return {
        one.id: [passage.id for passage in one.passages]
        for one in QuestionQueue().facts(topic_id)
    }


def test_a_bridge_is_offered_with_every_passage_it_rests_on(corpus, engine) -> None:
    """The anchor alone is what the verifier could never recover from."""
    written = corpus(topics=1, facts_per_topic=2, documents=2)
    topic_id = written["topics"][0]
    anchor, other = (one.passage_id for one in QuestionQueue().facts(topic_id))
    fact_id = _bridge(engine, anchor, other)

    assert _passages_of(engine, topic_id)[fact_id] == [anchor, other]


def test_a_bridge_carries_the_document_of_each_of_its_passages(corpus, engine) -> None:
    """Which is what makes it a cross-document question."""
    written = corpus(topics=1, facts_per_topic=2, documents=2)
    topic_id = written["topics"][0]
    anchor, other = (one.passage_id for one in QuestionQueue().facts(topic_id))
    fact_id = _bridge(engine, anchor, other)

    offered = next(one for one in QuestionQueue().facts(topic_id) if one.id == fact_id)
    assert len({passage.doc_sha256 for passage in offered.passages}) == 2


def test_a_bridge_that_recorded_no_citation_is_never_offered(corpus, engine) -> None:
    """It was drawn before the prompt said where in each passage it rests."""
    written = corpus(topics=1, facts_per_topic=2, documents=2)
    topic_id = written["topics"][0]
    anchor, other = (one.passage_id for one in QuestionQueue().facts(topic_id))
    fact_id = _bridge(engine, anchor, other, cited=False)

    assert fact_id not in _passages_of(engine, topic_id)


def test_a_bridge_left_resting_on_one_passage_is_never_offered(corpus, engine) -> None:
    """A claim no single passage states cannot rest on a single passage."""
    written = corpus(topics=1, facts_per_topic=2, documents=2)
    topic_id = written["topics"][0]
    anchor, _other = (one.passage_id for one in QuestionQueue().facts(topic_id))
    fact_id = _bridge(engine, anchor)

    assert fact_id not in _passages_of(engine, topic_id)


def _settings(**overrides):
    """The question settings a re-check reads, with nothing served."""
    from question_generation.config import Settings

    return Settings(
        per_topic=4,
        sample_size=4,
        fact_kinds=("atomic", "bridge"),
        type_mix={"factoid": 1},
        difficulty_mix={"easy": 1},
        followup_types=("condition",),
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
        **overrides,
    )


def test_a_re_check_leaves_a_bridge_backed_question_accepted(engine, database) -> None:
    """The scopes it was stored with are the ones the re-check reads back.

    A bridge rests on two passages through one fact. Counted the old way -
    one passage per fact - the re-check would read `single_passage` off a
    question stored `multi_passage` and reject every one of them as
    `source_changed` on its first run.
    """
    from question_generation.repository import QuestionCatalog
    from question_generation.service import reverify

    with Session(engine) as session:
        session.add_all([document(digest("a")), document(digest("b"))])
        session.flush()
        here = passage(digest("a"), ordinal=1, language="en")
        there = passage(digest("b"), ordinal=1, language="en")
        session.add_all([here, there])
        session.flush()
        spanning = fact(here.id, statement="Both kinds are answered on a clock.")
        session.add(spanning)
        session.flush()
        asked = question(
            question_text="Which kinds of request are answered on a clock?",
            target_answer="standard and urgent",
            status="accepted",
            difficulty="hard",
            passage_scope="multi_passage",
            document_scope="cross_document",
            topic_scope="single_topic",
            answer_chars=19,
        )
        session.add(asked)
        session.flush()
        session.add(QuestionFact(question_id=asked.id, fact_id=spanning.id))
        session.commit()
        asked_id, fact_id = asked.id, spanning.id
        anchor, other = here.id, there.id

    _bridge(engine, anchor, other)
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE facts SET kind = 'bridge' WHERE id = :id"), {"id": fact_id}
        )

    assert reverify(QuestionCatalog(), _settings()) == 0, (
        "the re-check rejected a question nothing had changed under"
    )
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT status, rejected_reason FROM questions WHERE id = :id"),
            {"id": asked_id},
        ).one()
    assert (row.status, row.rejected_reason) == ("accepted", None)


def test_a_bridge_backed_question_is_found_by_either_of_its_documents(
    engine, database
) -> None:
    """The page must not contradict the cross_document it was stored with."""
    from question_generation.repository import QuestionCatalog

    with Session(engine) as session:
        session.add_all([document(digest("a")), document(digest("b"))])
        session.flush()
        here = passage(digest("a"), ordinal=1, language="en")
        there = passage(digest("b"), ordinal=1, language="en")
        session.add_all([here, there])
        session.flush()
        spanning = fact(here.id, statement="Both kinds are answered on a clock.")
        session.add(spanning)
        session.flush()
        asked = question(question_text="Which kinds?", status="accepted")
        session.add(asked)
        session.flush()
        session.add(QuestionFact(question_id=asked.id, fact_id=spanning.id))
        session.commit()
        fact_id, anchor, other = spanning.id, here.id, there.id

    _bridge(engine, anchor, other)
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE facts SET kind = 'bridge' WHERE id = :id"), {"id": fact_id}
        )

    catalog = QuestionCatalog()
    assert catalog.page(document=digest("a"))[0] == 1
    assert catalog.page(document=digest("b"))[0] == 1, "its other half"
    # Unfiltered, so the aggregate sees every passage: a document filter
    # narrows the joined rows and so narrows this list too, which is how the
    # listing has always read.
    _, listed = catalog.page()
    assert sorted(listed[0].documents) == sorted([digest("a"), digest("b")])
