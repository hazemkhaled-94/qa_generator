"""The assessment queue against the database that runs it.

The unit tests prove what the judge is asked and what a verdict becomes.
These prove the half only PostgreSQL can answer: that enrolling twice
enrols once, that a claim reads back the text the templates need, that a
verdict and its metrics land together, and that deleting an artefact takes
its judgement with it.

That last one is the reason this stage has three foreign keys rather than
a polymorphic pair, and it is the only place that choice is checked.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, fitted, link, passage, question
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from assessment.models import Assessed, Judgement
from assessment.repository import AssessmentCatalog, AssessmentQueue
from database.qa_generator import (
    Assessment,
    AssessmentMetric,
    Fact,
    JudgeMetric,
    QuestionStatus,
    Status,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(engine, database):
    """One document, one passage, one fact, one topic and two questions.

    One of the questions is unanswerable, which is the row enrolment must
    leave alone: two of the three question metrics ask whether the facts
    support the answer, and it has none.
    """
    with Session(engine) as session:
        sha = digest("a")
        session.add(document(sha))
        session.flush()
        held = passage(sha)
        session.add(held)
        session.flush()
        one = fact(held.id, statement="A reply is due in five days.")
        session.add(one)
        session.add(fitted(0, label="Response times"))
        session.flush()

        answerable = question(
            question_text="How long is allowed for a reply?",
            target_answer="Five days.",
            status=QuestionStatus.ACCEPTED,
        )
        unanswerable = question(
            question_text="What is the budget?",
            answerable=False,
            status=QuestionStatus.ACCEPTED,
        )
        session.add_all([answerable, unanswerable])
        session.flush()
        session.add(link(answerable.id, one.id))
        session.add(link(unanswerable.id, one.id))
        session.commit()
        return {
            "fact": one.id,
            "question": answerable.id,
            "unanswerable": unanswerable.id,
        }


def enrolled(engine) -> dict[str, int]:
    """How many assessments of each kind there are."""
    with engine.connect() as connection:
        return dict(
            connection.execute(
                select(Assessment.artifact_kind, func.count()).group_by(
                    Assessment.artifact_kind
                )
            ).all()
        )


def test_start_enrols_every_artefact_and_queues_it(corpus, engine) -> None:
    """Nothing upstream creates these rows, so `start` is what does.

    A `start` that only requeued would do nothing at all the first time:
    there would be no row to move.
    """
    queued = AssessmentQueue().start()

    assert enrolled(engine) == {"fact": 1, "topic": 1, "question": 1}
    assert queued == 3


def test_an_unanswerable_question_is_not_enrolled(corpus, engine) -> None:
    """It has no answer for two of its three metrics to be about.

    Written deliberately, to test whether a chatbot admits it does not
    know. Judging it would report the pipeline's most careful output as
    its worst.
    """
    AssessmentQueue().start()

    with Session(engine) as session:
        judged = session.scalars(select(Assessment.question_id)).all()

    assert corpus["unanswerable"] not in judged
    assert corpus["question"] in judged


def test_enrolling_twice_enrols_once(corpus, engine) -> None:
    """So a re-run over a judged corpus costs one statement per kind."""
    queue = AssessmentQueue()
    queue.enrol()

    assert queue.enrol() == 0
    assert sum(enrolled(engine).values()) == 3


def test_a_claim_reads_back_the_text_its_templates_need(corpus) -> None:
    """The fields, by the names the templates substitute."""
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()

    claimed = queue.claim()

    assert claimed is not None
    assert claimed.kind == "fact"
    assert claimed.artifact_id == corpus["fact"]
    assert claimed.fields == {
        "statement": "A reply is due in five days.",
        "evidence": claimed.fields["evidence"],
    }


def test_a_question_is_claimed_with_the_facts_it_rests_on(corpus) -> None:
    """Which is what its three judgements are asked about."""
    queue = AssessmentQueue(kinds=("question",))
    queue.start()

    claimed = queue.claim()

    assert claimed is not None
    assert claimed.fields["question"] == "How long is allowed for a reply?"
    assert claimed.fields["answer"] == "Five days."
    assert "A reply is due in five days." in claimed.fields["facts"]


def test_a_topic_is_claimed_with_its_terms_and_its_name(corpus) -> None:
    """A request to refit is not a topic and is never claimed here."""
    queue = AssessmentQueue(kinds=("topic",))
    queue.start()

    claimed = queue.claim()

    assert claimed is not None
    assert claimed.fields["label"] == "Response times"
    assert "device" in claimed.fields["terms"]


def _judged(assessment_id: int, approved: bool) -> Assessed:
    """One artefact's verdict, as the service hands it over."""
    return Assessed(
        assessment_id=assessment_id,
        judgements=(
            Judgement(
                metric=JudgeMetric.HALLUCINATION,
                label="factual" if approved else "hallucinated",
                score=0.0 if approved else 1.0,
                approved=approved,
                explanation="the evidence says so",
            ),
            Judgement(
                metric=JudgeMetric.RELEVANCE,
                label="relevant",
                score=1.0,
                approved=True,
                explanation="on subject",
            ),
        ),
        judge_model="test/judge",
        prompt_version="1",
        trace_id="a" * 32,
        span_id="b" * 16,
    )


def test_a_verdict_and_its_metrics_land_together(corpus, engine) -> None:
    """One transaction: a verdict with no metrics says nothing about why."""
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()
    claimed = queue.claim()
    assert claimed is not None

    queue.record(_judged(claimed.assessment_id, approved=True))

    with Session(engine) as session:
        row = session.get(Assessment, claimed.assessment_id)
        assert row is not None
        assert row.approved is True
        assert row.assess_status == Status.ASSESSED
        assert row.judge_model == "test/judge"
        assert row.assessed_at is not None
        assert row.span_id == "b" * 16
        assert {one.metric for one in row.metrics} == {
            JudgeMetric.HALLUCINATION,
            JudgeMetric.RELEVANCE,
        }


def test_judging_again_replaces_the_metrics_rather_than_adding_to_them(
    corpus, engine
) -> None:
    """A corpus never holds two opinions with nothing saying which is current."""
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()
    claimed = queue.claim()
    assert claimed is not None

    queue.record(_judged(claimed.assessment_id, approved=True))
    queue.record(_judged(claimed.assessment_id, approved=False))

    with Session(engine) as session:
        held = session.scalars(select(func.count()).select_from(AssessmentMetric)).one()
        row = session.get(Assessment, claimed.assessment_id)

    assert held == 2
    assert row is not None
    assert row.approved is False


def test_deleting_a_fact_deletes_its_judgement(corpus, engine) -> None:
    """The reason this table has three foreign keys and not a (kind, id) pair.

    A judgement about a deleted fact is an opinion about nothing, and a
    polymorphic pair would have left it for a trigger to sweep.
    """
    AssessmentQueue(kinds=("fact",)).start()

    with Session(engine) as session:
        session.delete(session.get(Fact, corpus["fact"]))
        session.commit()

    assert enrolled(engine).get("fact", 0) == 0


def test_the_report_counts_a_disagreement_only_one_way(corpus, engine) -> None:
    """Kept by the pipeline and refused by the judge, and never the reverse.

    A fact the checker rejected and the judge approved is not a queue for
    anybody: most rejections are for something the judge was never asked
    about.
    """
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()
    claimed = queue.claim()
    assert claimed is not None
    queue.record(_judged(claimed.assessment_id, approved=False))

    quality = AssessmentCatalog().quality("fact")

    assert quality.judged == 1
    assert quality.approved == 0
    assert quality.disagreements == 1, "the seeded fact is validated"


def test_verdicts_are_read_back_by_artefact_id(corpus) -> None:
    """Which is what the export and the Argilla push both need."""
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()
    claimed = queue.claim()
    assert claimed is not None
    queue.record(_judged(claimed.assessment_id, approved=True))

    found = AssessmentCatalog().verdicts_for("fact", [corpus["fact"]])

    assert set(found) == {corpus["fact"]}
    assert found[corpus["fact"]].approved is True
    assert len(found[corpus["fact"]].metrics) == 2


def test_a_page_across_the_kinds_is_one_window_and_not_three(corpus) -> None:
    """Every row once, in id order, however many kinds are in the page.

    The three kinds reach their artefacts through three different foreign
    keys, so there is no one join that serves all of them. Paging each
    separately and slicing afterwards is a different query: page two would
    be each kind's second page, which skips rows nobody asked to skip and
    repeats rows already shown.
    """
    AssessmentQueue().start()
    catalog = AssessmentCatalog()

    total, everything = catalog.page(limit=None)
    _, first = catalog.page(limit=2, offset=0)
    _, second = catalog.page(limit=2, offset=2)

    assert total == 3
    assert [one.id for one in everything] == sorted(
        (one.id for one in everything), reverse=True
    )
    seen = [one.id for one in first] + [one.id for one in second]
    assert seen == [one.id for one in everything]
    assert len(set(seen)) == 3, "a row appeared on two pages"


def test_the_queue_narrows_to_one_kind(corpus, engine) -> None:
    """How a deployment pays for the questions and not for the facts."""
    queue = AssessmentQueue()
    queue.enrol()

    moved = queue.start(queue.narrowed("kind", "question"))

    with Session(engine) as session:
        pending = dict(
            session.execute(
                select(Assessment.artifact_kind, Assessment.assess_status)
            ).all()
        )

    assert moved == 1
    assert pending["question"] == Status.PENDING
    assert pending["fact"] == Status.NEW


def test_a_search_looks_in_the_artefact_and_in_what_the_judge_said(corpus) -> None:
    """Two places, and the second is the half nothing else here can search.

    Until this phase there were no model opinions stored, so "which answers
    did it call unsupported, and in what words" had no query at all.
    """
    queue = AssessmentQueue(kinds=("fact",))
    queue.start()
    claimed = queue.claim()
    assert claimed is not None
    queue.record(_judged(claimed.assessment_id, approved=False))
    catalog = AssessmentCatalog()

    assert catalog.page(search="five days", field="artifact")[0] == 1
    assert catalog.page(search="five days", field="explanation")[0] == 0
    assert catalog.page(search="evidence says", field="explanation")[0] == 1
    assert catalog.page(search="evidence says", field="both")[0] == 1
    assert catalog.page(search="nothing like this")[0] == 0


def test_a_percent_sign_somebody_typed_is_not_a_wildcard(corpus) -> None:
    """`matching` autoescapes, which a hand-rolled ILIKE would not."""
    AssessmentQueue(kinds=("fact",)).start()

    assert AssessmentCatalog().page(search="%")[0] == 0


def test_a_queue_told_nothing_enrols_what_the_settings_say(corpus, engine) -> None:
    """The api builds one of these with no arguments at all.

    It serves `POST /assessment/enrol` and `POST /assessment/start`, and a
    default of "every kind, every row" is not a default - it is the
    deployment's cost decision being ignored. Against a real corpus that
    default enrolled 11,392 artefacts under an ASSESSMENT_SAMPLE of 200,
    and `start` would have queued all of them.
    """
    AssessmentQueue().enrol()

    assert sum(enrolled(engine).values()) == 3, (
        "the fixture holds one artefact of each judgeable kind, and "
        "ASSESSMENT_SAMPLE in the test environment is well above that"
    )


def test_what_a_caller_names_beats_what_the_settings_say(corpus, engine) -> None:
    """The worker resolves both to stamp the version beside them."""
    AssessmentQueue(kinds=("question",), sample=1).enrol()

    assert set(enrolled(engine)) == {"question"}
