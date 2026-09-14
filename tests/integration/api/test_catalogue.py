"""Reading back what the stages produced, and the platform's own routes."""

from __future__ import annotations

import pytest
from seed import digest, document, fact, passage
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def produced(client, engine):
    """One document, two passages and two facts, as a run would leave them."""
    with Session(engine) as session:
        session.add(document(digest("a"), language="en", chunk_status="chunked"))
        session.flush()
        first = passage(
            digest("a"), ordinal=1, text="The device weighs 4 kg.", language="en"
        )
        second = passage(
            digest("a"), ordinal=2, text="It ships in March.", block_type="table"
        )
        session.add_all([first, second])
        session.flush()
        session.add_all(
            [
                fact(first.id, statement="The device weighs 4 kg."),
                fact(
                    first.id,
                    statement="The device weighs 7 kg.",
                    validated=False,
                    rejection_code="unsupported_addition",
                    validation_error="the statement asserts 7",
                ),
            ]
        )
        session.commit()
        return client, first.id


def test_the_passages_listing_pages_and_totals(produced) -> None:
    """What the Passages page draws."""
    client, _ = produced

    listed = client.get("/passages").json()

    assert listed["total"] == 2
    assert {one["ordinal"] for one in listed["passages"]} == {1, 2}


def test_the_passages_listing_can_be_narrowed_to_one_document(produced) -> None:
    """Every page offers the same document picker."""
    client, _ = produced

    listed = client.get("/passages", params={"document": digest("a")}).json()

    assert listed["total"] == 2


def test_a_document_with_no_passages_lists_none(produced) -> None:
    """An empty answer, not an error."""
    client, _ = produced

    listed = client.get("/passages", params={"document": digest("nothing")}).json()

    assert listed["total"] == 0
    assert listed["passages"] == []


def test_the_block_types_are_offered_as_a_filter(produced) -> None:
    """Drawn from what the corpus actually holds."""
    client, _ = produced

    assert "table" in client.get("/passages/types").json()


def test_one_passage_is_served_in_full(produced) -> None:
    """Where a citation is traced back to the text it names."""
    client, passage_id = produced

    served = client.get(f"/passages/{passage_id}")

    assert served.status_code == 200
    body = served.json()
    assert body["passage"]["text"] == "The device weighs 4 kg."
    assert body["extract_status"] == "new"
    assert body["table_cells"] == [] and body["sentences"] == []


def test_the_listing_carries_counts_rather_than_the_grids(produced) -> None:
    """A page of fifty would otherwise ship every cell of every table."""
    client, _ = produced

    one = client.get("/passages").json()["passages"][0]

    assert "table_cells" not in one
    assert "sentences" not in one


def test_a_passage_that_does_not_exist_is_a_404(produced) -> None:
    """The refusal the page turns into a message."""
    client, _ = produced

    answered = client.get("/passages/999999")

    assert answered.status_code == 404
    assert answered.json()["code"]


def test_the_facts_listing_reports_both_verdicts(produced) -> None:
    """A rejected fact is kept, so the report can group on why."""
    client, _ = produced

    listed = client.get("/facts").json()

    assert listed["total"] == 2
    assert {one["validated"] for one in listed["facts"]} == {True, False}


def test_a_refused_fact_is_listed_with_the_code_that_refused_it(produced) -> None:
    """Kept rather than discarded, so a report can group on why."""
    client, _ = produced

    refused = [
        one for one in client.get("/facts").json()["facts"] if not one["validated"]
    ]

    assert len(refused) == 1
    assert refused[0]["rejection_code"] == "unsupported_addition"
    assert refused[0]["validation_error"]


def test_the_facts_listing_can_be_narrowed_to_one_document(produced) -> None:
    """The same document picker every page offers."""
    client, _ = produced

    assert client.get("/facts", params={"document": digest("a")}).json()["total"] == 2
    assert (
        client.get("/facts", params={"document": digest("none")}).json()["total"] == 0
    )


def test_the_facts_listing_can_be_searched(produced) -> None:
    """Trigram search over the statement."""
    client, _ = produced

    listed = client.get("/facts", params={"q": "7 kg"}).json()

    assert listed["total"] == 1


def test_the_quality_figures_are_reported(produced) -> None:
    """What the Facts page's top row draws."""
    client, _ = produced

    quality = client.get("/facts/quality").json()

    assert isinstance(quality, dict)
    assert quality


def test_health_touches_no_dependency(client) -> None:
    """What the container healthcheck calls."""
    answered = client.get("/health")

    assert answered.status_code == 200
    assert answered.json() == {"status": "ok"}


def test_status_reports_every_component(client) -> None:
    """The deep check, behind the health panel."""
    answered = client.get("/status")

    assert answered.status_code == 200
    body = answered.json()
    assert {"database", "object_store"} <= set(body)
    assert body["database"]["ok"] is True
    assert body["object_store"]["ok"] is True


def test_status_names_each_service_it_fronts(client) -> None:
    """The panel renders whatever is there without knowing the names."""
    body = client.get("/status").json()

    assert {
        "ingestion",
        "parsing",
        "chunking",
        "extraction",
        "topic_modelling",
    } <= set(body)
