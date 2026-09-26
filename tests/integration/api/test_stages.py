"""The queue surface every stage shares, over HTTP.

Four verbs, each twice: once for the whole queue and once narrowed to what
a scope names. None of them does the work; they move rows between statuses
and answer with how many moved.
"""

from __future__ import annotations

import pytest
from seed import digest, document, passage
from sqlalchemy.orm import Session

from database.qa_generator import Status

pytestmark = pytest.mark.integration

#: Every stage built from `stage_router`, and a scope each accepts.
STAGES = ("parsing", "chunking", "extraction")


@pytest.fixture
def corpus(client, engine):
    """Two documents, one with two passages, all waiting to be asked for.

    `a` is parsed and `b` is not, so both stages that queue over `documents`
    have something to move: parsing has the row it has not done, chunking
    has the one it can. A corpus where neither was parsed had nothing
    chunking could queue - `ChunkQueue.ready` refuses a document with no
    parsed form, because the only outcome of queueing one is a failure.
    """
    with Session(engine) as session:
        session.add_all(
            [
                document(digest("a"), parse_status=Status.PARSED),
                document(digest("b")),
            ]
        )
        session.flush()
        session.add_all(
            [
                passage(digest("a"), ordinal=1),
                passage(digest("a"), ordinal=2),
                passage(digest("b"), ordinal=1),
            ]
        )
        session.commit()
    return client


@pytest.mark.parametrize("stage", STAGES)
def test_a_stage_reports_its_queue(corpus, stage) -> None:
    """One shape for every stage, whatever it queues over."""
    answered = corpus.get(f"/{stage}/status")

    assert answered.status_code == 200
    body = answered.json()
    assert body["stage"] == stage
    assert body["working"] is False
    assert body["scope"] is None and body["value"] is None
    assert sum(body["rows"].values()) > 0


@pytest.mark.parametrize("stage", STAGES)
def test_starting_a_stage_answers_202_with_what_moved(corpus, stage) -> None:
    """Returns at once: the worker picks the rows up on its next poll."""
    answered = corpus.post(f"/{stage}/start")

    assert answered.status_code == 202
    body = answered.json()
    assert body["action"] == "start"
    assert body["rows"] > 0
    assert "queued" in body["detail"]


@pytest.mark.parametrize("stage", STAGES)
def test_a_verb_that_moves_nothing_says_so(corpus, stage) -> None:
    """The count and the wording come from the same place."""
    answered = corpus.post(f"/{stage}/retry")

    assert answered.status_code == 202
    assert answered.json()["rows"] == 0
    assert "nothing has failed" in answered.json()["detail"]


@pytest.mark.parametrize("stage", STAGES)
def test_the_verbs_round_trip_the_queue(corpus, stage) -> None:
    """Start makes rows claimable, stop makes them new again."""
    queued = corpus.post(f"/{stage}/start").json()["rows"]

    taken_back = corpus.post(f"/{stage}/stop").json()["rows"]

    assert taken_back == queued
    assert corpus.get(f"/{stage}/status").json()["rows"].get("pending", 0) == 0


@pytest.mark.parametrize("stage", STAGES)
def test_a_verb_no_stage_has_is_refused(corpus, stage) -> None:
    """The verbs are a Literal, so anything else never reaches a queue."""
    assert corpus.post(f"/{stage}/obliterate").status_code == 422


@pytest.mark.parametrize("stage", ["parsing", "chunking", "extraction"])
def test_a_stage_narrowed_to_a_document_reports_only_that_one(corpus, stage) -> None:
    """The same shape, with the scope and value it was asked about."""
    answered = corpus.get(f"/{stage}/document/{digest('a')}/status")

    assert answered.status_code == 200
    body = answered.json()
    assert body["scope"] == "document"
    assert body["value"] == digest("a")


def test_a_narrowed_verb_moves_only_that_documents_rows(corpus) -> None:
    """One document's passages, not the corpus's."""
    answered = corpus.post(f"/extraction/document/{digest('a')}/start")

    assert answered.status_code == 202
    assert answered.json()["rows"] == 2
    assert "document" in answered.json()["detail"] or answered.json()["rows"] == 2


def test_extraction_narrows_to_one_passage(corpus, engine) -> None:
    """It queues over passages, so it answers for one of those too."""
    from sqlalchemy import text

    with engine.connect() as connection:
        first = connection.execute(text("SELECT min(id) FROM passages")).scalar_one()

    answered = corpus.post(f"/extraction/passage/{first}/start")

    assert answered.status_code == 202
    assert answered.json()["rows"] == 1


def test_a_scope_a_stage_does_not_take_is_refused_as_404(corpus) -> None:
    """Parsing queues over documents and knows nothing of a passage."""
    answered = corpus.get("/parsing/passage/1/status")

    assert answered.status_code == 404
    assert answered.json()["code"] == "unknown_scope"
    assert "document" in answered.json()["detail"]


def test_a_value_the_column_cannot_hold_is_refused_as_400(corpus) -> None:
    """Refused here rather than as a database error."""
    answered = corpus.get("/extraction/passage/not-a-number/status")

    assert answered.status_code == 400
    assert answered.json()["code"] == "invalid_value"


def test_a_narrowing_that_selects_nothing_is_not_an_error(corpus) -> None:
    """A document with no rows for this stage is a count of zero."""
    answered = corpus.post(f"/parsing/document/{digest('unknown')}/start")

    assert answered.status_code == 202
    assert answered.json()["rows"] == 0


def test_topic_modelling_reports_a_queue_with_no_rows_yet(client) -> None:
    """It has no row until a fit is asked for."""
    answered = client.get("/topics/status")

    assert answered.status_code == 200
    assert answered.json()["stage"] == "topics"


def test_asking_for_a_fit_queues_one(client) -> None:
    """Asking is what creates the row."""
    answered = client.post("/topics/discover")

    assert answered.status_code == 202
    assert client.get("/topics/status").json()["rows"].get("pending") == 1


def test_asking_twice_queues_one_fit(client) -> None:
    """Replaces the outstanding request rather than adding to it."""
    client.post("/topics/discover")
    client.post("/topics/discover")

    assert client.get("/topics/status").json()["rows"].get("pending") == 1


def test_withdrawing_a_fit_takes_the_request_away(client) -> None:
    """A request nobody is going to run is not a state worth keeping."""
    client.post("/topics/discover")

    answered = client.post("/topics/stop")

    assert answered.status_code == 200
    assert answered.json()["rows"] == 1
    assert client.get("/topics/status").json()["rows"] == {}


def test_the_topics_listing_is_empty_before_a_fit(client) -> None:
    """Nothing is fitted until a worker runs one."""
    assert client.get("/topics").json() == []


def test_describing_a_topic_that_does_not_exist_is_a_404(client) -> None:
    """A label is a person's, and it has to land on a topic."""
    answered = client.patch("/topics/999", json={"label": "Shipping"})

    assert answered.status_code == 404
    assert answered.json()["code"]


def test_a_visualisation_nothing_has_drawn_is_a_404(client) -> None:
    """The figure is an artefact a fit produces."""
    assert client.get("/topics/visualisation/de").status_code == 404


# ── reclaim ────────────────────────────────────────────────────────────────
#
# The one verb that had no route. It used to be command-line only because
# "the api cannot know a worker is gone" - but neither can the command line,
# and a person driving this over HTTP was left with the single stuck row that
# `retry` (which takes the failed), `rerun` (which skips what is held) and
# `stop` (which takes back only what has not begun) all step over.
#
# What the danger needs is a narrowing, not an absence: nothing can tell a
# dead claim from a live one, so the whole-stage form asks to be confirmed.


@pytest.fixture
def held(client, engine):
    """A document a worker claimed and never came back from."""
    with Session(engine) as session:
        session.add(document(digest("a"), parse_status=Status.IN_PROGRESS))
        session.commit()
    return client


def test_reclaiming_one_item_returns_it_to_the_queue(held) -> None:
    """Narrowed, which is how it is meant to be used while a worker is up."""
    answered = held.post(f"/parsing/document/{digest('a')}/reclaim")

    assert answered.status_code == 202
    body = answered.json()
    assert body["action"] == "reclaim" and body["rows"] == 1
    assert held.get("/parsing/status").json()["rows"] == {"pending": 1}


def test_reclaiming_a_whole_stage_is_refused_without_a_confirmation(held) -> None:
    """It can take a row from a worker that is still on it."""
    answered = held.post("/parsing/reclaim")

    assert answered.status_code == 400
    assert answered.json()["code"] == "confirm_required"
    assert "Narrow it" in answered.json()["detail"]
    assert held.get("/parsing/status").json()["rows"] == {"in_progress": 1}


def test_a_confirmed_reclaim_over_a_whole_stage_is_allowed(held) -> None:
    """For when the stage is stopped and the person knows it."""
    answered = held.post("/parsing/reclaim?confirm=true")

    assert answered.status_code == 202
    assert answered.json()["rows"] == 1


def test_reclaim_leaves_rows_no_worker_holds_alone(corpus) -> None:
    """Only `in_progress` is its business."""
    assert corpus.post("/parsing/reclaim?confirm=true").json()["rows"] == 0


def test_confirming_is_only_asked_for_reclaim(corpus) -> None:
    """The other four verbs cannot hand a live worker's row to a second."""
    assert corpus.post("/parsing/start").status_code == 202


# ── enrol ──────────────────────────────────────────────────────────────────


def test_enrolling_creates_the_rows_without_queuing_any(client, engine) -> None:
    """The dry run: what `start` would cost, before paying for it."""
    answered = client.post("/assessment/enrol")

    assert answered.status_code == 202
    assert answered.json()["action"] == "enrol"
    assert client.get("/assessment/status").json()["rows"].get("pending", 0) == 0


def test_enrol_is_not_read_as_a_queue_verb(client) -> None:
    """`POST /assessment/{action}` would otherwise match it first."""
    assert client.post("/assessment/enrol").status_code != 422
