"""How one artefact was produced, over HTTP.

A corpus of one document, one passage, one fact, one topic and one
question, so the chain a question reports is the whole pipeline and every
edge in it is one this walks.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, fitted, membership, passage
from sqlalchemy.orm import Session

from database.qa_generator import Question as QuestionRow
from database.qa_generator import QuestionFact, Status

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(engine, database) -> dict:
    """One of each artefact, joined the way the pipeline joins them."""
    sha = digest("a")
    with Session(engine) as session:
        session.add(document(sha, title="A report", parse_status=Status.PARSED))
        held = fitted(0, label="Delivery")
        session.add(held)
        session.flush()

        at = passage(sha, ordinal=1, text="Requests are answered in 48 hours.")
        session.add(at)
        session.flush()
        session.add(membership(at.id, held.id))

        drawn = fact(at.id, statement="Requests are answered within 48 hours.")
        session.add(drawn)
        session.flush()

        asked = QuestionRow(
            question_text="How long do requests take?",
            target_answer="48 hours",
            answerable=True,
            language="en",
            status="rejected",
            rejected_reason="leaks_source",
            gates_ran=["structural", "near_duplicate", "phrasing"],
        )
        session.add(asked)
        session.flush()
        session.add(QuestionFact(question_id=asked.id, fact_id=drawn.id))
        found = {
            "document": sha,
            "passage": at.id,
            "fact": drawn.id,
            "topic": held.id,
            "question": asked.id,
        }
        session.commit()
    return found


def test_a_question_reports_the_whole_pipeline_in_order(client, corpus) -> None:
    """Five steps, numbered by the stage that produced each."""
    answered = client.get(f"/lineage/question/{corpus['question']}")

    assert answered.status_code == 200
    chain = answered.json()
    assert [(one["position"], one["kind"]) for one in chain["steps"]] == [
        (2, "document"),
        (3, "passage"),
        (4, "fact"),
        (5, "topic"),
        (6, "question"),
    ]
    assert [one["stage"] for one in chain["steps"]] == [
        "parsing",
        "chunking",
        "extraction",
        "topic_modelling",
        "question_generation",
    ]


def test_each_step_carries_the_artefact_it_produced(client, corpus) -> None:
    """The ids are the rows the corpus was built from."""
    chain = client.get(f"/lineage/question/{corpus['question']}").json()
    at = {one["kind"]: one for one in chain["steps"]}

    assert at["document"]["artifacts"][0]["id"] == corpus["document"]
    assert at["document"]["artifacts"][0]["label"] == "A report"
    assert at["fact"]["artifacts"][0]["id"] == str(corpus["fact"])
    assert at["topic"]["artifacts"][0]["label"] == "Delivery"
    assert at["question"]["artifacts"][0]["verdict"] == "rejected"
    assert at["question"]["artifacts"][0]["reason"] == "leaks_source"


def test_a_question_reports_every_gate_that_read_it(client, corpus) -> None:
    """At its fixed position, the one that refused it last and failed."""
    chain = client.get(f"/lineage/question/{corpus['question']}").json()

    assert chain["gates_recorded"] is True
    assert [
        (one["position"], one["name"], one["passed"]) for one in chain["gates"]
    ] == [
        (1, "structural", True),
        (4, "near_duplicate", True),
        (5, "phrasing", False),
    ]


def test_a_document_is_the_root_of_its_own_chain(client, corpus) -> None:
    """Nothing produced it, so it is the only step and there are no gates."""
    chain = client.get(f"/lineage/document/{corpus['document']}").json()

    assert [one["kind"] for one in chain["steps"]] == ["document"]
    assert chain["gates"] == []
    assert chain["gates_recorded"] is False


def test_a_fact_reaches_its_passage_its_document_and_its_topic(client, corpus) -> None:
    """Upwards to the two, and sideways to the topic its passage sits in."""
    chain = client.get(f"/lineage/fact/{corpus['fact']}").json()

    assert [one["kind"] for one in chain["steps"]] == [
        "document",
        "passage",
        "fact",
        "topic",
    ]


def test_a_step_reports_its_total_beside_what_it_lists(client, corpus) -> None:
    """A cap that hid a fan-out silently would be worse than no cap."""
    chain = client.get(f"/lineage/topic/{corpus['topic']}").json()
    at = {one["kind"]: one for one in chain["steps"]}

    assert at["passage"]["total"] == len(at["passage"]["artifacts"]) == 1


def test_an_id_nothing_holds_is_a_404(client, corpus) -> None:
    """Rather than an empty chain, which reads as an artefact with no past."""
    assert client.get("/lineage/fact/99999999").status_code == 404


def test_an_id_that_is_not_a_number_names_nothing(client, corpus) -> None:
    """Every kind but a document is keyed by a row id, and int() raises."""
    assert client.get("/lineage/fact/not-a-number").status_code == 404
    assert client.get("/lineage/question/:").status_code == 404


def test_a_kind_the_pipeline_does_not_produce_is_refused(client) -> None:
    """The route names what may be traced, so the OpenAPI document does."""
    assert client.get("/lineage/sentence/1").status_code == 422
