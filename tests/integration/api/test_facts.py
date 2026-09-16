"""The HTTP surface over the facts, against the real wiring.

What the Facts page reads: one listing narrowed four ways, the figures under
the same filter, and the passages a bridge rests on.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, passage
from sqlalchemy.orm import Session

from database.qa_generator import FactKind, FactPassage, Rejection

pytestmark = pytest.mark.integration


@pytest.fixture
def produced(client, engine):
    """Two documents, and one fact of each kind across them."""
    with Session(engine) as session:
        for seed in ("a", "b"):
            session.add(document(digest(seed), language="en", chunk_status="chunked"))
        session.flush()
        first = passage(digest("a"), ordinal=1, text="Answered within 48 hours.")
        second = passage(digest("b"), ordinal=1, text="Raised by phone.")
        session.add_all([first, second])
        session.flush()

        bridge = fact(first.id, statement="Both name a way in.", kind=FactKind.BRIDGE)
        session.add_all(
            [
                fact(first.id, statement="Answered within 48 hours."),
                fact(
                    first.id,
                    statement="A short summary of the passage.",
                    kind=FactKind.SUMMARY,
                ),
                fact(
                    first.id,
                    statement="- A point\n- Another point",
                    kind=FactKind.OUTLINE,
                    validated=False,
                    rejection_code=Rejection.NOT_CONDENSED,
                    validation_error="the digest is 24 characters against 25",
                ),
                fact(
                    second.id,
                    statement="Phone - Line: open",
                    extraction_method="deterministic",
                ),
                bridge,
            ]
        )
        session.flush()
        session.add_all(
            [
                FactPassage(fact_id=bridge.id, passage_id=first.id, position=0),
                FactPassage(fact_id=bridge.id, passage_id=second.id, position=1),
            ]
        )
        session.commit()
        return client, {"bridge": bridge.id, "first": first.id, "second": second.id}


def test_every_kind_is_listed_with_the_value_it_was_stored_under(produced) -> None:
    """The page renders whatever is there without knowing the names."""
    client, _ = produced

    listed = client.get("/facts").json()

    assert listed["total"] == 5
    assert {one["kind"] for one in listed["facts"]} == {
        FactKind.ATOMIC,
        FactKind.SUMMARY,
        FactKind.OUTLINE,
        FactKind.BRIDGE,
    }


@pytest.mark.parametrize(
    ("kind", "found"),
    [("atomic", 2), ("summary", 1), ("outline", 1), ("bridge", 1)],
)
def test_the_listing_narrows_to_one_kind(produced, kind, found) -> None:
    """Each reading of the corpus is looked at on its own."""
    client, _ = produced

    listed = client.get("/facts", params={"kind": kind}).json()

    assert listed["total"] == found
    assert {one["kind"] for one in listed["facts"]} == {kind}


def test_a_kind_nothing_writes_is_refused_before_it_reaches_the_queue(
    produced,
) -> None:
    """A Literal, so the OpenAPI document lists what is accepted."""
    client, _ = produced

    assert client.get("/facts", params={"kind": "paragraph"}).status_code == 422


def test_the_kind_filter_narrows_the_figures_too(produced) -> None:
    """One filter, so the report and the rows cannot disagree."""
    client, _ = produced

    quality = client.get("/facts/quality", params={"kind": "atomic"}).json()

    assert quality["total"] == 2
    assert quality["kinds"] == {"atomic": 2}


def test_the_quality_report_counts_each_reading(produced) -> None:
    """What the page puts beside the headline figures."""
    client, _ = produced

    quality = client.get("/facts/quality").json()

    assert quality["kinds"] == {"atomic": 2, "summary": 1, "outline": 1, "bridge": 1}
    assert quality["total"] == 5
    assert quality["validated"] == 4


def test_a_refused_digest_is_grouped_on_its_code(produced) -> None:
    """A message carrying a measurement would give one bucket per fact."""
    client, _ = produced

    quality = client.get("/facts/quality").json()

    assert quality["rejected"] == {Rejection.NOT_CONDENSED: 1}


def test_the_filters_combine(produced) -> None:
    """Every figure below is narrowed by all of them at once."""
    client, _ = produced

    listed = client.get(
        "/facts",
        params={"document": digest("a"), "kind": "atomic", "method": "llm"},
    ).json()

    assert listed["total"] == 1
    assert listed["facts"][0]["statement"] == "Answered within 48 hours."


def test_a_bridge_names_the_passages_it_rests_on(produced) -> None:
    """Anchor first, in the order the model was shown them."""
    client, ids = produced

    answered = client.get(f"/facts/{ids['bridge']}/passages").json()

    assert answered["fact"] == ids["bridge"]
    assert answered["passages"] == [ids["first"], ids["second"]]


def test_another_kind_rests_on_no_group(produced) -> None:
    """The listing already names the one passage it rests on."""
    client, _ = produced
    atomic = next(
        one
        for one in client.get("/facts", params={"kind": "atomic"}).json()["facts"]
        if one["validated"]
    )

    answered = client.get(f"/facts/{atomic['id']}/passages").json()

    assert answered["passages"] == []


def test_a_fact_that_does_not_exist_rests_on_nothing(produced) -> None:
    """A page asking about a deleted fact gets an answer, not a failure."""
    client, _ = produced

    answered = client.get("/facts/999999/passages")

    assert answered.status_code == 200
    assert answered.json()["passages"] == []


def test_the_quality_route_is_not_read_as_a_fact_id(produced) -> None:
    """`/facts/quality` is a route, and `quality` is not a number."""
    client, _ = produced

    assert client.get("/facts/quality").status_code == 200
