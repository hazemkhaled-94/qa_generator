"""Ingest, against the database and the object store it writes to.

The unit tests cover what an upload's bytes say about it. These cover what
the service does with two stores that can each fail independently.
"""

from __future__ import annotations

import pytest
from seed import digest
from sqlalchemy import text

from database.qa_generator import Outcome
from ingestion.models import UploadedFile
from ingestion.repository import DocumentRepository
from ingestion.service import IngestService

pytestmark = pytest.mark.integration

#: A one-page PDF, small enough to keep in the file and real enough to read.
PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)


@pytest.fixture
def service(database, buckets):
    """The service as the composition root wires it."""
    from blob_store.seaweedfs import DocumentsBucket

    return IngestService(
        repository=DocumentRepository(),
        store=DocumentsBucket(),
        max_file_size_bytes=1024 * 1024,
        allowed_media_types=("application/pdf",),
        pipeline_version="0.0.1",
    )


def rows(engine, table: str) -> int:
    """Counts what one table holds."""
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def test_an_accepted_upload_writes_the_object_and_the_row(service, engine) -> None:
    """Object before row: an object with no row is harmless."""
    result = service.ingest(UploadedFile("report.pdf", PDF))

    assert result.outcome == Outcome.STORED
    assert result.sha256 is not None
    assert rows(engine, "documents") == 1
    assert rows(engine, "ingest_events") == 1
    assert service.stored_file(result.sha256) == (PDF, "application/pdf")


def test_the_same_bytes_twice_are_stored_once(service, engine) -> None:
    """Duplicate detection is byte-exact, and cheaper than reading the pages."""
    first = service.ingest(UploadedFile("report.pdf", PDF))
    second = service.ingest(UploadedFile("another-name.pdf", PDF))

    assert second.outcome == Outcome.DUPLICATE_BYTES
    assert second.sha256 == first.sha256
    assert rows(engine, "documents") == 1
    assert rows(engine, "ingest_events") == 2, "every attempt is recorded"


def test_an_upload_over_the_limit_is_refused_and_recorded(service, engine) -> None:
    """Refused before the pages are read, and before anything is stored."""
    result = service.ingest(
        UploadedFile("big.pdf", b"%PDF-" + b"x" * (2 * 1024 * 1024))
    )

    assert result.outcome == Outcome.TOO_LARGE
    assert "exceeds" in (result.detail or "")
    assert rows(engine, "documents") == 0
    assert rows(engine, "ingest_events") == 1


def test_a_refusal_on_the_declared_size_reads_no_bytes(service, engine) -> None:
    """What the route does with Content-Length, before touching the body."""
    result = service.refuse_oversized("huge.pdf", 99 * 1024 * 1024)

    assert result.outcome == Outcome.TOO_LARGE
    assert result.sha256 is None
    assert rows(engine, "documents") == 0
    assert rows(engine, "ingest_events") == 1


def test_something_that_is_not_a_document_is_refused(service, engine) -> None:
    """The type comes from the leading bytes, not from the name."""
    result = service.ingest(UploadedFile("report.pdf", b"GIF89a not a pdf"))

    assert result.outcome == Outcome.UNSUPPORTED_TYPE
    assert rows(engine, "documents") == 0
    assert rows(engine, "ingest_events") == 1


def test_a_pdf_nothing_can_read_is_refused(service, engine) -> None:
    """The magic bytes are necessary and not sufficient."""
    result = service.ingest(UploadedFile("broken.pdf", b"%PDF-1.4\ntruncated"))

    assert result.outcome == Outcome.UNSUPPORTED_TYPE
    assert rows(engine, "documents") == 0


def test_the_metadata_is_enough_to_rebuild_the_row(service) -> None:
    """The bucket alone can say what a document was and when it arrived."""
    from urllib.parse import unquote

    from blob_store.seaweedfs import DocumentsBucket, s3_client

    result = service.ingest(UploadedFile("Jahresbericht.pdf", PDF))
    assert result.sha256 is not None

    key = DocumentsBucket.key_for(result.sha256, "application/pdf")
    held = s3_client().head_object(Bucket="documents", Key=key)["Metadata"]
    written = {name: unquote(value) for name, value in held.items()}

    assert written["sha256"] == result.sha256
    assert written["original-filename"] == "Jahresbericht.pdf"
    assert written["pipeline-version"] == "0.0.1"
    assert written["ingested-at"]


def test_a_row_that_cannot_be_written_takes_its_object_back_out(
    database, buckets, engine, monkeypatch
) -> None:
    """Nothing would reference the object and nothing would collect it."""
    from blob_store.seaweedfs import DocumentsBucket

    bucket = DocumentsBucket()
    built = IngestService(
        repository=DocumentRepository(),
        store=bucket,
        max_file_size_bytes=1024 * 1024,
        allowed_media_types=("application/pdf",),
        pipeline_version="0.0.1",
    )
    monkeypatch.setattr(
        DocumentRepository,
        "store",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no room")),
    )

    with pytest.raises(RuntimeError, match="no room"):
        built.ingest(UploadedFile("report.pdf", PDF))

    upload = UploadedFile("report.pdf", PDF)
    assert bucket.find(bucket.key_for(upload.sha256, "application/pdf")) is None
    assert rows(engine, "documents") == 0


def test_an_unknown_digest_never_reaches_the_store(service) -> None:
    """Looked up first, so a path parameter cannot become an object key."""
    assert service.stored_file(digest("z")) is None


def test_the_catalogue_lists_what_was_ingested(service) -> None:
    """What the Documents page reads."""
    service.ingest(UploadedFile("first.pdf", PDF))
    service.ingest(UploadedFile("second.pdf", PDF.replace(b"200 200", b"300 300")))

    total, listed = service.documents()

    assert total == 2
    assert {one.filename for one in listed} == {"first.pdf", "second.pdf"}


def test_the_catalogue_can_be_searched_and_paged(service) -> None:
    """One page at a time, with the total behind it."""
    service.ingest(UploadedFile("annual-report.pdf", PDF))
    service.ingest(
        UploadedFile("something-else.pdf", PDF.replace(b"200 200", b"300 300"))
    )

    total, listed = service.documents("annual", limit=10, offset=0)

    assert total == 1
    assert [one.filename for one in listed] == ["annual-report.pdf"]


def test_a_page_beyond_the_end_is_empty_but_counted(service) -> None:
    """The total is of the search, not of the page."""
    service.ingest(UploadedFile("first.pdf", PDF))

    total, listed = service.documents(limit=10, offset=10)

    assert total == 1
    assert listed == []


def test_every_document_is_offered_to_a_picker(service) -> None:
    """The cheapest listing, because three pages poll it."""
    service.ingest(UploadedFile("first.pdf", PDF))

    named = service.document_names()

    assert [one.filename for one in named] == ["first.pdf"]
