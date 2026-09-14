"""Deleting a document, across the three stores that hold one.

Nothing is atomic across two stores, so what matters is the order and what
survives each half of it. Irreversible either way, which is why every path
is exercised here rather than by hand.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, link, passage, question
from sqlalchemy import text
from sqlalchemy.orm import Session

from ingestion.factory import build_removal

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7\nnot really a document\n"


@pytest.fixture
def service(database, buckets):
    """The removal service as the composition root wires it."""
    return build_removal()


@pytest.fixture
def held(engine, buckets):
    """A document with a file, a converted form and derived rows."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket

    sha = digest()
    source, parsed = DocumentsBucket(), ParsedBucket()
    source.put(
        source.key_for(sha, "application/pdf"), PDF, content_type="application/pdf"
    )
    parsed.put(parsed.key_for(sha), b"{}", content_type="application/json")

    with Session(engine) as session:
        session.add(document(sha, chunk_status="chunked"))
        session.flush()
        first = passage(sha, ordinal=1)
        second = passage(sha, ordinal=2)
        session.add_all([first, second])
        session.flush()
        drawn = fact(first.id)
        session.add(drawn)
        session.flush()
        asked = question()
        session.add(asked)
        session.flush()
        session.add(link(asked.id, drawn.id))
        session.commit()
    return sha


def counts(engine) -> dict[str, int]:
    """What each table holds."""
    with engine.connect() as connection:
        return {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar_one()
            for table in ("documents", "passages", "facts", "questions")
        }


def test_deleting_a_document_removes_every_trace_of_it(service, held, engine) -> None:
    """The rows, the file and the converted form."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket

    removed = service.delete(held)

    assert removed is not None
    assert removed.document is True
    assert removed.passages == 2
    assert removed.file is True
    assert removed.parsed is True
    assert counts(engine) == {
        "documents": 0,
        "passages": 0,
        "facts": 0,
        "questions": 0,
    }
    assert (
        DocumentsBucket().find(DocumentsBucket.key_for(held, "application/pdf")) is None
    )
    assert ParsedBucket().find(ParsedBucket.key_for(held)) is None


def test_deleting_an_unknown_document_answers_nothing(service) -> None:
    """What the route turns into a 404."""
    assert service.delete(digest("z")) is None


def test_a_document_with_no_converted_form_still_deletes(
    service, database, buckets, engine
) -> None:
    """Parsing may never have run; the file alone is enough to remove."""
    from blob_store.seaweedfs import DocumentsBucket

    sha = digest("b")
    source = DocumentsBucket()
    source.put(
        source.key_for(sha, "application/pdf"), PDF, content_type="application/pdf"
    )
    with Session(engine) as session:
        session.add(document(sha))
        session.commit()

    removed = service.delete(sha)

    assert removed is not None
    assert removed.file is True
    assert removed.parsed is False, "there was nothing converted to remove"
    assert counts(engine)["documents"] == 0


def test_a_row_whose_object_is_already_gone_still_deletes(
    service, database, buckets, engine
) -> None:
    """A half-finished earlier deletion must not block the next one."""
    sha = digest("c")
    with Session(engine) as session:
        session.add(document(sha))
        session.commit()

    removed = service.delete(sha)

    assert removed is not None
    assert removed.file is False
    assert removed.document is True
    assert counts(engine)["documents"] == 0


def test_deleting_derived_data_keeps_the_document_and_its_files(
    service, held, engine
) -> None:
    """The next run rebuilds the passages without re-parsing."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket

    removed = service.delete_derived(held)

    assert removed is not None
    assert removed.document is False
    assert removed.passages == 2
    assert removed.file is False and removed.parsed is False
    assert counts(engine) == {
        "documents": 1,
        "passages": 0,
        "facts": 0,
        "questions": 0,
    }
    assert DocumentsBucket().find(DocumentsBucket.key_for(held, "application/pdf"))
    assert ParsedBucket().find(ParsedBucket.key_for(held))


def test_dropping_derived_data_returns_chunking_to_the_queue(
    service, held, engine
) -> None:
    """Chunking goes back to `new`, so nothing runs until somebody asks."""
    service.delete_derived(held)

    with engine.connect() as connection:
        status = connection.execute(
            text("SELECT chunk_status FROM documents")
        ).scalar_one()

    assert status == "new"


def test_dropping_derived_data_of_an_unknown_document_answers_nothing(
    service,
) -> None:
    """The same refusal as a full delete."""
    assert service.delete_derived(digest("z")) is None


def test_deleting_one_document_leaves_the_other_alone(
    service, held, database, buckets, engine
) -> None:
    """The corpus is not the unit of deletion."""
    other = digest("d")
    with Session(engine) as session:
        session.add(document(other))
        session.flush()
        session.add(passage(other, ordinal=1))
        session.commit()

    service.delete(held)

    assert counts(engine) == {
        "documents": 1,
        "passages": 1,
        "facts": 0,
        "questions": 0,
    }


def test_the_upload_record_survives_a_deletion(service, held, database, engine) -> None:
    """ingest_events is the history of what was submitted, not of what is held."""
    from seed import event

    with Session(engine) as session:
        session.add(event(sha256=held))
        session.commit()

    service.delete(held)

    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT count(*) FROM ingest_events")).scalar_one()
            == 1
        )
