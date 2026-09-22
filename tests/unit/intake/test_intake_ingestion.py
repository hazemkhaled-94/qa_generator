"""What the ingest flow writes, refuses and takes back out.

`test_uploaded_file` covers what an upload's bytes say about it. These cover
what the service does with those answers, and with two stores that can each
fail independently.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from intake_drivers import (
    NOT_A_PDF,
    PDF,
    IngestDriver,
    ReaderSpy,
    RemovalDriver,
)

from database.qa_generator import Outcome, Status
from ingestion.models import PdfFacts
from ingestion.pdf import UnreadablePdf


def test_an_allowlist_this_build_cannot_detect_is_refused_at_start_up() -> None:
    """Support that does not exist must not read as support."""
    with pytest.raises(ValueError, match="image/tiff"):
        IngestDriver(allowed=("application/pdf", "image/tiff"))

    with pytest.raises(ValueError, match="Detectable: application/pdf"):
        IngestDriver(allowed=("image/tiff",))


def test_the_upload_limit_is_published_so_a_caller_can_refuse_one_unread() -> None:
    """What the route checks Content-Length against."""
    assert IngestDriver(max_file_size_bytes=4096).service.max_file_size_bytes == 4096


def test_an_accepted_upload_writes_the_object_and_the_row() -> None:
    """Both, and the outcome names what happened."""
    driver = IngestDriver()

    result = driver.upload("report.pdf")

    assert result.outcome == Outcome.STORED
    assert result.detail is None
    assert driver.stored == [result.sha256]
    assert driver.objects == [driver.store.key_for(result.sha256, "application/pdf")]
    assert driver.outcomes == [Outcome.STORED]


def test_the_stored_object_carries_the_bytes_and_the_detected_type() -> None:
    """Not the declared type, and not the extension."""
    driver = IngestDriver()

    result = driver.upload("report.txt")

    key = driver.store.key_for(result.sha256, "application/pdf")
    assert driver.store.objects[key] == PDF
    assert driver.store.content_types[key] == "application/pdf"


def test_the_metadata_is_enough_to_rebuild_the_row() -> None:
    """The bucket alone can say what a document was and when it arrived."""
    driver = IngestDriver(pipeline_version="9.9.9")

    result = driver.upload("Jahresbericht.pdf")
    assert result.sha256

    written = driver.metadata_of(result.sha256)

    assert written["sha256"] == result.sha256
    assert written["original-filename"] == "Jahresbericht.pdf"
    assert written["pipeline-version"] == "9.9.9"
    assert datetime.fromisoformat(written["ingested-at"]).tzinfo is not None


def test_nothing_mutable_is_written_into_the_metadata() -> None:
    """S3 metadata cannot change without rewriting the object."""
    driver = IngestDriver()
    result = driver.upload()
    assert result.sha256

    assert set(driver.metadata_of(result.sha256)) == {
        "sha256",
        "original-filename",
        "ingested-at",
        "pipeline-version",
    }


def test_the_same_bytes_twice_are_stored_once_and_recorded_twice() -> None:
    """Duplicate detection is byte-exact, and every attempt is a row."""
    driver = IngestDriver()

    first = driver.upload("report.pdf")
    second = driver.upload("another-name.pdf")

    assert second.outcome == Outcome.DUPLICATE_BYTES
    assert second.sha256 == first.sha256
    assert len(driver.stored) == 1
    assert driver.outcomes == [Outcome.STORED, Outcome.DUPLICATE_BYTES]


def test_a_duplicate_is_never_read_as_a_pdf(monkeypatch) -> None:
    """Hashing bytes in memory is cheaper than walking every page."""
    reader = ReaderSpy()
    driver = IngestDriver(reader=reader, monkeypatch=monkeypatch)

    driver.upload("report.pdf")
    driver.upload("again.pdf")

    assert reader.calls == 1, "the second upload read the document again"


def test_the_counts_the_reader_took_are_what_the_row_holds(monkeypatch) -> None:
    """Parsing reads these back to decide whether a document is scanned."""
    reader = ReaderSpy(PdfFacts(page_count=73, char_count=91_000))
    driver = IngestDriver(reader=reader, monkeypatch=monkeypatch)

    result = driver.upload()
    assert result.sha256

    held = driver.rows.documents[result.sha256]
    assert (held["page_count"], held["char_count"]) == (73, 91_000)


def test_an_upload_at_the_limit_is_accepted() -> None:
    """The limit is what is allowed, not what is refused."""
    driver = IngestDriver(max_file_size_bytes=len(PDF))

    assert driver.upload().outcome == Outcome.STORED


def test_an_upload_one_byte_over_the_limit_is_refused() -> None:
    """And the refusal names both figures."""
    driver = IngestDriver(max_file_size_bytes=len(PDF) - 1)

    result = driver.upload()

    assert result.outcome == Outcome.TOO_LARGE
    assert "exceeds" in (result.detail or "")
    assert "MB limit" in (result.detail or "")


def test_an_oversized_upload_is_neither_read_nor_stored(monkeypatch) -> None:
    """Refused before the pages are read, and before anything is written."""
    reader = ReaderSpy()
    driver = IngestDriver(max_file_size_bytes=1, reader=reader, monkeypatch=monkeypatch)

    driver.upload()

    assert reader.calls == 0
    assert driver.stored == []
    assert driver.objects == []
    assert driver.outcomes == [Outcome.TOO_LARGE]


def test_a_refusal_on_the_declared_size_writes_only_the_event() -> None:
    """With no bytes there is no digest and no document."""
    driver = IngestDriver(max_file_size_bytes=1024)

    result = driver.refuse("huge.pdf", 99 * 1024 * 1024)

    assert result.outcome == Outcome.TOO_LARGE
    assert result.sha256 is None
    assert "99.0 MB exceeds the 0 MB limit" in (result.detail or "")
    assert driver.stored == []
    assert driver.rows.events[0]["sha256"] is None


@pytest.mark.parametrize(
    ("data", "detected"),
    [
        (NOT_A_PDF, "unknown"),
        (b"", "unknown"),
        (b"%PD", "unknown"),
        (b"\x00\x00%PDF-1.7", "unknown"),
    ],
)
def test_something_no_magic_number_claims_is_refused(
    data: bytes, detected: str
) -> None:
    """The type comes from the leading bytes, and the refusal says so."""
    driver = IngestDriver()

    result = driver.upload("report.pdf", data)

    assert result.outcome == Outcome.UNSUPPORTED_TYPE
    assert detected in (result.detail or "")
    assert driver.stored == []
    assert driver.objects == []


def test_a_detectable_type_absent_from_the_allowlist_is_refused() -> None:
    """The allowlist is the policy; detection only says what it is."""
    driver = IngestDriver(allowed=())

    result = driver.upload()

    assert result.outcome == Outcome.UNSUPPORTED_TYPE
    assert "application/pdf" in (result.detail or "")


def test_a_pdf_nothing_can_read_is_refused_with_the_reason(monkeypatch) -> None:
    """The magic bytes are necessary and not sufficient."""
    driver = IngestDriver(
        reader=ReaderSpy(raises=UnreadablePdf("the PDF is password protected")),
        monkeypatch=monkeypatch,
    )

    result = driver.upload("locked.pdf")

    assert result.outcome == Outcome.UNSUPPORTED_TYPE
    assert result.detail == "the PDF is password protected"
    assert driver.stored == []
    assert driver.objects == [], "an unreadable document left an object behind"


def test_a_row_that_cannot_be_written_takes_its_object_back_out() -> None:
    """Nothing would reference the object and nothing would collect it."""
    driver = IngestDriver()
    driver.rows.refuses_store = RuntimeError("no room")

    with pytest.raises(RuntimeError, match="no room"):
        driver.upload()

    assert driver.objects == []
    assert driver.stored == []
    assert len(driver.store.removed) == 1


def test_an_object_store_that_refuses_writes_no_row() -> None:
    """Object before row, so the failure stops before the row."""
    driver = IngestDriver()
    driver.store.refuses_put = RuntimeError("bucket is gone")

    with pytest.raises(RuntimeError, match="bucket is gone"):
        driver.upload()

    assert driver.stored == []


def test_losing_the_race_to_insert_answers_duplicate_and_keeps_the_object() -> None:
    """The key is content-addressed, so the object the loser wrote is the same."""
    driver = IngestDriver()
    driver.rows.raced = True

    result = driver.upload()

    assert result.outcome == Outcome.DUPLICATE_BYTES
    assert result.sha256 is not None
    assert driver.objects != [], "the object was taken back out of a shared key"


def test_a_stored_document_is_read_back_exactly() -> None:
    """Byte for byte, with the type it was detected as."""
    driver = IngestDriver()
    result = driver.upload()
    assert result.sha256

    assert driver.service.stored_file(result.sha256) == (PDF, "application/pdf")


def test_an_unknown_digest_never_reaches_the_store() -> None:
    """Looked up first, so a path parameter cannot become an object key."""
    driver = IngestDriver()
    driver.store.refuses_get = AssertionError("the store was asked")

    assert driver.service.stored_file("f" * 64) is None


# ── Removal ────────────────────────────────────────────────────────────────

SHA = "a" * 64
OTHER = "b" * 64


def test_deleting_a_document_removes_every_trace_of_it() -> None:
    """Its rows, its file and its converted form."""
    driver = RemovalDriver(held=(SHA,), passages=7)

    removed = driver.delete(SHA)

    assert removed is not None
    assert (removed.document, removed.file, removed.parsed) == (True, True, True)
    assert removed.passages == 7
    assert driver.rows.documents == {}
    assert not driver.holds_file(SHA)
    assert not driver.holds_parsed(SHA)


def test_the_objects_are_archived_rather_than_deleted() -> None:
    """The file leaves its bucket and is still readable in the archive."""
    driver = RemovalDriver(held=(SHA,))

    driver.delete(SHA)

    assert sorted(driver.archive.objects) == [
        f"documents/{driver.store.key_for(SHA, 'application/pdf')}",
        f"parsed/{driver.parsed.key_for(SHA)}",
    ]
    assert driver.store.removed == [], "the bucket's own delete would not have kept it"


def test_deleting_an_unknown_document_answers_nothing_and_touches_no_store() -> None:
    """The digest is looked up before it becomes an object key."""
    driver = RemovalDriver()

    assert driver.delete(SHA) is None
    assert driver.archive.objects == {}
    assert driver.store.removed == []
    assert driver.parsed.removed == []


def test_a_document_whose_file_is_already_gone_still_deletes() -> None:
    """Reported per store rather than assumed."""
    driver = RemovalDriver(held=(SHA,))
    driver.store.objects.clear()

    removed = driver.delete(SHA)

    assert removed is not None
    assert removed.file is False
    assert removed.document is True


def test_a_document_with_no_converted_form_still_deletes() -> None:
    """A document that never parsed has nothing in the parsed bucket."""
    driver = RemovalDriver(held=(SHA,))
    driver.parsed.objects.clear()

    removed = driver.delete(SHA)

    assert removed is not None
    assert removed.parsed is False


def test_the_objects_go_before_the_row(monkeypatch) -> None:
    """A row pointing at a missing object fails every later stage."""
    driver = RemovalDriver(held=(SHA,))
    monkeypatch.setattr(
        driver.rows,
        "delete",
        lambda sha: (_ for _ in ()).throw(RuntimeError("the database went away")),
    )

    with pytest.raises(RuntimeError, match="went away"):
        driver.delete(SHA)

    assert not driver.holds_file(SHA), "the row was deleted before its object"
    assert not driver.holds_parsed(SHA)


def test_deleting_one_document_leaves_the_other_alone() -> None:
    """Nothing here is corpus-wide."""
    driver = RemovalDriver(held=(SHA, OTHER), passages=2)

    driver.delete(SHA)

    assert list(driver.rows.documents) == [OTHER]
    assert driver.holds_file(OTHER)
    assert driver.holds_parsed(OTHER)


def test_dropping_the_derived_data_keeps_the_document_and_both_objects() -> None:
    """The next run rebuilds the passages without re-parsing."""
    driver = RemovalDriver(held=(SHA,), passages=4)

    removed = driver.delete_derived(SHA)

    assert removed is not None
    assert (removed.document, removed.file, removed.parsed) == (False, False, False)
    assert removed.passages == 4
    assert SHA in driver.rows.documents
    assert driver.holds_file(SHA)
    assert driver.holds_parsed(SHA)


def test_dropping_the_derived_data_returns_chunking_to_not_started() -> None:
    """`new` rather than `pending`: starting a stage stays a decision."""
    driver = RemovalDriver(held=(SHA,), passages=4)

    driver.delete_derived(SHA)

    assert driver.rows.chunk_status[SHA] == Status.NEW


def test_dropping_the_derived_data_of_an_unknown_document_answers_nothing() -> None:
    """And writes nothing."""
    driver = RemovalDriver()

    assert driver.delete_derived(SHA) is None
    assert driver.rows.chunk_status == {}


# ── Reading the PDF ────────────────────────────────────────────────────────


def built_pdf(pages: int = 1, **saved) -> bytes:
    """Builds a real PDF with one line of text on each page."""
    import pymupdf

    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"Page {number} of this document.")
    try:
        return document.tobytes(**saved)
    finally:
        document.close()


def test_the_pages_and_the_characters_are_counted() -> None:
    """The two figures parsing decides `scanned` from."""
    from ingestion.pdf import read_pdf

    facts = read_pdf(built_pdf(pages=3))

    assert facts.page_count == 3
    assert facts.char_count >= 3 * len("Page 0 of this document.")


def test_the_characters_are_summed_page_by_page() -> None:
    """So a long document's text is never all held at once."""
    from ingestion.pdf import read_pdf

    one = read_pdf(built_pdf(pages=1)).char_count
    three = read_pdf(built_pdf(pages=3)).char_count

    assert one > 0
    assert three == 3 * one


def test_a_pdf_with_no_text_layer_counts_no_characters() -> None:
    """Which is what a scanned document looks like from here."""
    import pymupdf

    from ingestion.pdf import read_pdf

    document = pymupdf.open()
    document.new_page()
    try:
        empty = document.tobytes()
    finally:
        document.close()

    facts = read_pdf(empty)

    assert (facts.page_count, facts.char_count) == (1, 0)


def test_a_password_protected_pdf_is_refused_by_that_name() -> None:
    """Named as what it is, so the uploader can do something about it."""
    import pymupdf

    from ingestion.pdf import UnreadablePdf, read_pdf

    locked = built_pdf(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u"
    )

    with pytest.raises(UnreadablePdf, match="password protected"):
        read_pdf(locked)


def test_a_pdf_that_cannot_be_opened_carries_the_readers_reason() -> None:
    """The magic bytes are necessary and not sufficient."""
    from ingestion.pdf import UnreadablePdf, read_pdf

    with pytest.raises(UnreadablePdf, match="could not be read"):
        read_pdf(b"%PDF-1.4\ntruncated")


def test_the_reader_is_not_what_decides_a_file_is_a_pdf() -> None:
    """Bytes no magic number claims open as an empty document, not an error.

    The leading bytes are the type check, and they run first. This is what
    would be stored if they ever stopped.
    """
    from ingestion.pdf import read_pdf

    assert read_pdf(NOT_A_PDF).char_count == 0
