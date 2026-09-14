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

from database.qa_generator import Status
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
    from question_generation.models import CheckedQuestion

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    facts = tuple(written["facts"][topic_id])
    QuestionQueue().start()
    QuestionQueue().claim()

    stored = QuestionQueue().store(
        topic_id,
        [
            CheckedQuestion(
                question_text="What does the device weigh?",
                target_answer="4 kg",
                answerable=True,
                difficulty="cross_passage",
                language="en",
                status="accepted",
                rejected_reason=None,
                fact_ids=facts,
                embedding=[0.1] * 1024,
            )
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
    from question_generation.models import CheckedQuestion

    written = corpus(topics=1, facts_per_topic=2)
    topic_id = written["topics"][0]
    facts = written["facts"][topic_id]
    queue = QuestionQueue()

    for position, fact_id in enumerate(facts):
        queue.store(
            topic_id,
            [
                CheckedQuestion(
                    question_text=f"Question {position}?",
                    target_answer="4 kg",
                    answerable=True,
                    difficulty="single_passage",
                    language="en",
                    status="accepted",
                    rejected_reason=None,
                    fact_ids=(fact_id,),
                )
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
