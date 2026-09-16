"""The three repositories, against the database that runs them.

`database/test_queue.py` covers the queue mechanics every stage shares. These
cover what is this service's own: which columns each stage writes, which ones
it must leave alone, what a re-chunk takes with it, and what the catalogue
reads back.
"""

from __future__ import annotations

import pytest
from seed import digest, document, event, fact, passage
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.qa_generator import Status
from ingestion.repository import DocumentRepository
from preprocessing.chunking.models import Chunk, Chunking
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from preprocessing.parsing.models import ParsedDocument
from preprocessing.parsing.repository import ParseQueue

pytestmark = pytest.mark.integration

A, B = digest("a"), digest("b")


def chunk(
    ordinal: int = 1,
    text_content: str = "The device weighs 4 kg.",
    **columns,
) -> Chunk:
    """One passage as the builder produced it."""
    return Chunk(
        ordinal=ordinal,
        text=text_content,
        page_from=columns.pop("page_from", 1),
        page_to=columns.pop("page_to", 1),
        section_path=columns.pop("section_path", "1 > 1.2 Delivery"),
        block_type=columns.pop("block_type", "text"),
        doc_item_refs=columns.pop("doc_item_refs", ["#/texts/0"]),
        bbox=columns.pop("bbox", [{"page": 1, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}]),
        table_cells=columns.pop("table_cells", []),
        language=columns.pop("language", "en"),
        sentences=columns.pop(
            "sentences", [{"i": 0, "start": 0, "end": 23, "predicates": 1}]
        ),
        lemmas=columns.pop("lemmas", ["device", "kg"]),
    )


def one(engine, statement: str, **params):
    """Reads one row back."""
    with engine.connect() as connection:
        return connection.execute(text(statement), params).one()


def rows(engine, statement: str, **params):
    """Reads every row back."""
    with engine.connect() as connection:
        return connection.execute(text(statement), params).all()


# ── Parsing's half of the document row ─────────────────────────────────────


@pytest.fixture
def parsable(engine, database):
    """One document pending parsing, with the counts ingestion took."""
    with Session(engine) as session:
        session.add(
            document(
                A,
                parse_status=Status.PENDING,
                page_count=73,
                char_count=91_000,
                parse_error="a failure from an earlier run",
            )
        )
        session.commit()
    return A


PARSED = ParsedDocument(
    title="Annual Report",
    language="de",
    content_sha256=digest("content"),
    page_count=71,
    confidence=0.88,
    confidence_low=0.71,
)


def test_a_claim_carries_what_the_converter_needs_to_dispatch(parsable) -> None:
    """The media type and the two counts, all written at ingest."""
    claimed = ParseQueue().claim()

    assert claimed is not None
    assert claimed.sha256 == A
    assert claimed.media_type == "application/pdf"
    assert (claimed.page_count, claimed.char_count) == (73, 91_000)


def test_a_claim_on_a_document_with_no_counts_carries_none(engine, database) -> None:
    """Both columns are nullable, and parsing reads them as scanned."""
    with Session(engine) as session:
        session.add(document(A, parse_status=Status.PENDING))
        session.commit()

    claimed = ParseQueue().claim()

    assert claimed is not None
    assert (claimed.page_count, claimed.char_count) == (None, None)


def test_completing_a_parse_writes_the_content_columns(parsable, engine) -> None:
    """Title, language, content digest and both confidences."""
    queue = ParseQueue()
    queue.claim()

    queue.complete(A, PARSED)

    row = one(
        engine,
        "SELECT title, language, content_sha256, parse_confidence, "
        "parse_confidence_low, parse_status, parse_error, parse_claimed_at "
        "FROM documents",
    )
    assert row.title == "Annual Report"
    assert row.language == "de"
    assert row.content_sha256 == PARSED.content_sha256
    assert (row.parse_confidence, row.parse_confidence_low) == (0.88, 0.71)
    assert row.parse_status == Status.PARSED
    assert row.parse_error is None, "an error from an earlier run survived"
    assert row.parse_claimed_at is None


def test_completing_a_parse_leaves_ingestions_page_count_alone(
    parsable, engine
) -> None:
    """Two stages count pages; the column belongs to the one that uploaded."""
    queue = ParseQueue()
    queue.claim()

    queue.complete(A, PARSED)

    assert one(engine, "SELECT page_count FROM documents").page_count == 73


def test_completing_a_parse_does_not_set_chunking_going(parsable, engine) -> None:
    """A stage knows of no other stage, so a re-parse re-chunks nothing."""
    queue = ParseQueue()
    queue.claim()

    queue.complete(A, PARSED)

    assert one(engine, "SELECT chunk_status FROM documents").chunk_status == Status.NEW


def test_another_document_holding_the_same_text_is_found(engine, database) -> None:
    """Different bytes, the same text: only knowable once the text exists."""
    with Session(engine) as session:
        session.add(document(A, content_sha256=digest("content")))
        session.add(document(B))
        session.commit()

    assert ParseQueue().holder_of(digest("content"), besides=B) == A


def test_a_document_is_never_reported_as_holding_its_own_text(engine, database) -> None:
    """Re-parsing one document must not read as a duplicate of itself."""
    with Session(engine) as session:
        session.add(document(A, content_sha256=digest("content")))
        session.commit()

    assert ParseQueue().holder_of(digest("content"), besides=A) is None


def test_text_nothing_holds_yet_has_no_holder(engine, database) -> None:
    """Which is the answer that lets the parse through."""
    with Session(engine) as session:
        session.add(document(A))
        session.commit()

    assert ParseQueue().holder_of(digest("content"), besides=A) is None


# ── Chunking's half of the document row ────────────────────────────────────


@pytest.fixture
def chunkable(engine, database):
    """One parsed document pending chunking, in German."""
    with Session(engine) as session:
        session.add(
            document(
                A,
                parse_status=Status.PARSED,
                language="de",
                chunk_status=Status.PENDING,
                chunk_error="a failure from an earlier run",
            )
        )
        session.commit()
    return A


def test_a_claim_carries_the_documents_language_as_a_fallback(chunkable) -> None:
    """A passage too short to judge is read under it."""
    claimed = ChunkQueue().claim()

    assert claimed is not None
    assert (claimed.sha256, claimed.language) == (A, "de")


def test_the_passages_the_oversized_count_and_the_status_land_together(
    chunkable, engine
) -> None:
    """One transaction, so nothing reads as chunked holding other passages."""
    stored = ChunkQueue().replace(
        A, Chunking(passages=[chunk(1), chunk(2)], oversized=1)
    )

    assert stored == 2
    row = one(
        engine,
        "SELECT chunk_status, chunk_error, chunk_claimed_at, oversized FROM documents",
    )
    assert row.chunk_status == Status.CHUNKED
    assert row.chunk_error is None
    assert row.chunk_claimed_at is None
    assert row.oversized == 1
    assert rows(engine, "SELECT ordinal FROM passages ORDER BY ordinal") == [(1,), (2,)]


def test_every_column_the_builder_read_reaches_the_row(chunkable, engine) -> None:
    """The passage a fact is cited against, in full."""
    ChunkQueue().replace(A, Chunking(passages=[chunk()], oversized=0))

    row = one(
        engine,
        "SELECT text, language, sentences, lemmas, page_from, page_to, "
        "section_path, block_type, doc_item_refs, bbox, table_cells, "
        "extract_status FROM passages",
    )
    assert row.text == "The device weighs 4 kg."
    assert row.language == "de" or row.language == "en"
    assert row.sentences == [{"i": 0, "start": 0, "end": 23, "predicates": 1}]
    assert row.lemmas == ["device", "kg"]
    assert (row.page_from, row.page_to) == (1, 1)
    assert row.section_path == "1 > 1.2 Delivery"
    assert row.block_type == "text"
    assert row.doc_item_refs == ["#/texts/0"]
    assert row.bbox == [{"b": 4.0, "l": 1.0, "r": 3.0, "t": 2.0, "page": 1}]
    assert row.table_cells is None, "an empty grid was stored as JSON null"
    assert row.extract_status == Status.NEW, "chunking must not queue extraction"


def test_a_passage_with_no_position_stores_null_rather_than_json_null(
    chunkable, engine
) -> None:
    """Otherwise every passage answers IS NOT NULL whether it has one or not."""
    ChunkQueue().replace(
        A, Chunking(passages=[chunk(bbox=[], sentences=[])], oversized=0)
    )

    row = one(engine, "SELECT bbox, sentences FROM passages")
    assert (row.bbox, row.sentences) == (None, None)


def test_re_chunking_replaces_every_passage_and_renumbers_from_one(
    chunkable, engine
) -> None:
    """A passage is immutable; a re-chunk is a new set."""
    queue = ChunkQueue()
    queue.replace(A, Chunking(passages=[chunk(1), chunk(2), chunk(3)], oversized=0))

    queue.replace(A, Chunking(passages=[chunk(1, "Only one now.")], oversized=0))

    assert rows(engine, "SELECT ordinal, text FROM passages") == [(1, "Only one now.")]


def test_re_chunking_takes_the_facts_drawn_from_the_old_passages(
    chunkable, engine
) -> None:
    """Which is why dropping the derived data is a separate decision."""
    queue = ChunkQueue()
    queue.replace(A, Chunking(passages=[chunk()], oversized=0))
    with Session(engine) as session:
        first = session.execute(text("SELECT min(id) FROM passages")).scalar_one()
        session.add(fact(first))
        session.commit()
    assert one(engine, "SELECT count(*) AS held FROM facts").held == 1

    queue.replace(A, Chunking(passages=[chunk()], oversized=0))

    assert one(engine, "SELECT count(*) AS held FROM facts").held == 0


def test_re_chunking_one_document_leaves_another_alone(engine, database) -> None:
    """Nothing here is corpus-wide."""
    with Session(engine) as session:
        session.add_all(
            [
                document(A, chunk_status=Status.PENDING),
                document(B, chunk_status=Status.CHUNKED),
            ]
        )
        session.add(passage(B, ordinal=1, text="Another document's passage."))
        session.commit()

    ChunkQueue().replace(A, Chunking(passages=[chunk()], oversized=0))

    assert (
        one(
            engine, "SELECT count(*) AS held FROM passages WHERE doc_sha256 = :b", b=B
        ).held
        == 1
    )


def test_the_stage_reports_what_it_owns(chunkable, engine) -> None:
    """The status panel's figures for this stage."""
    ChunkQueue().replace(A, Chunking(passages=[chunk(1), chunk(2)], oversized=0))

    counts = ChunkQueue().counts()

    assert counts["passages"] == 2
    assert counts[Status.CHUNKED] == 1


def test_a_passage_language_that_is_not_a_code_is_refused(chunkable, engine) -> None:
    """The column takes a two-letter code or NULL, and nothing between."""
    with pytest.raises(IntegrityError, match="passages_language_is_iso_639_1"):
        ChunkQueue().replace(A, Chunking(passages=[chunk(language="")], oversized=0))


def test_a_passage_with_no_detectable_language_is_stored(chunkable, engine) -> None:
    """NULL is what puts it outside every topic model, and is allowed."""
    ChunkQueue().replace(A, Chunking(passages=[chunk(language=None)], oversized=0))

    assert one(engine, "SELECT language FROM passages").language is None


# ── The passage catalogue ──────────────────────────────────────────────────


@pytest.fixture
def corpus(engine, database):
    """Two documents' passages, of three kinds and two languages."""
    with Session(engine) as session:
        session.add_all(
            [
                document(A, language="de", chunk_status=Status.CHUNKED),
                document(B, language="en", chunk_status=Status.CHUNKED),
            ]
        )
        session.add_all(
            [
                passage(
                    A,
                    ordinal=1,
                    text="Die Institute melden ihre Kennzahlen.",
                    section_path="1 Meldewesen",
                    block_type="text",
                    language="de",
                    sentences=[{"i": 0, "start": 0, "end": 36, "predicates": 1}],
                    lemmas=["institut", "kennzahl"],
                ),
                passage(
                    A,
                    ordinal=2,
                    text="| Modell | Masse |\n| Kompakt | 4 |",
                    section_path="2 Tabellen",
                    block_type="table",
                    language="de",
                    table_cells=[
                        {
                            "caption": "Tabelle 1",
                            "num_rows": 2,
                            "num_cols": 2,
                            "cells": [
                                {"row": 1, "col": 0, "text": "Kompakt", "line": 1}
                            ],
                        }
                    ],
                    sentences=[{"i": 0, "start": 0, "end": 18, "predicates": 0}],
                ),
                passage(
                    B,
                    ordinal=1,
                    text="The institutions report their figures.",
                    section_path="1 Reporting",
                    block_type="section_header",
                    language="en",
                    extract_status=Status.EXTRACTED,
                ),
            ]
        )
        session.commit()
    return PassageCatalog()


def test_the_listing_reports_the_total_behind_the_page(corpus) -> None:
    """Both in one call, so the count cannot describe other rows."""
    total, listed = corpus.page(limit=2, offset=0)

    assert total == 3
    assert len(listed) == 2


def test_a_page_beyond_the_end_is_empty_but_still_counted(corpus) -> None:
    """The total is of the filter, not of the page."""
    total, listed = corpus.page(limit=10, offset=10)

    assert (total, listed) == (3, [])


def test_the_listing_is_ordered_by_document_then_reading_order(corpus) -> None:
    """So a reader can follow a document through it."""
    _, listed = corpus.page()

    assert [(one.doc_sha256, one.ordinal) for one in listed] == sorted(
        (one.doc_sha256, one.ordinal) for one in listed
    )


def test_the_listing_can_be_narrowed_to_one_document(corpus) -> None:
    """Which is what the picker on the Passages page selects."""
    total, listed = corpus.page(document=B)

    assert total == 1
    assert [one.doc_sha256 for one in listed] == [B]


def test_the_listing_can_be_narrowed_to_one_block_type(corpus) -> None:
    """A table is read by a different extractor, so it is worth seeing alone."""
    total, listed = corpus.page(block_type="table")

    assert total == 1
    assert listed[0].ordinal == 2


def test_a_search_looks_in_the_column_it_was_told_to(corpus) -> None:
    """Nothing outside it is searched."""
    assert corpus.page(search="Meldewesen", field="text")[0] == 0
    assert corpus.page(search="Meldewesen", field="section")[0] == 1
    assert corpus.page(search="Meldewesen", field="both")[0] == 1


def test_a_search_with_no_column_named_looks_in_the_text(corpus) -> None:
    """Which is the column the toolbar offers first."""
    assert corpus.page(search="Institute", field=None)[0] == 1


def test_a_search_is_characters_and_not_a_pattern(corpus) -> None:
    """Whatever a person types reaches the database escaped."""
    assert corpus.page(search="%")[0] == 0
    assert corpus.page(search="_nstitute")[0] == 0


def test_the_filters_narrow_together_rather_than_separately(corpus) -> None:
    """Every figure on the page is of everything matching at once."""
    assert corpus.page(document=A, block_type="table", search="Modell")[0] == 1
    assert corpus.page(document=B, block_type="table")[0] == 0


def test_the_listing_carries_counts_rather_than_the_grids(corpus) -> None:
    """A page of fifty passages would otherwise ship every cell."""
    _, listed = corpus.page(document=A)

    table = next(one for one in listed if one.block_type == "table")
    assert (table.table_count, table.sentence_count) == (1, 1)
    assert not hasattr(table, "table_cells")


def test_one_passage_is_served_in_full(corpus, engine) -> None:
    """Its cell grids, its numbered units and where extraction has got to."""
    with engine.connect() as connection:
        held = connection.execute(
            text("SELECT id FROM passages WHERE block_type = 'table'")
        ).scalar_one()

    detail = corpus.passage(held)

    assert detail is not None
    assert detail.table_cells[0]["caption"] == "Tabelle 1"
    assert detail.sentences == [{"i": 0, "start": 0, "end": 18, "predicates": 0}]
    assert detail.extract_status == Status.NEW
    assert detail.extract_error is None


def test_a_passage_that_does_not_exist_answers_nothing(corpus) -> None:
    """Which the route turns into a 404."""
    assert corpus.passage(999_999) is None


def test_only_the_block_types_the_corpus_holds_are_offered(corpus) -> None:
    """A filter must not offer a value that selects nothing."""
    assert corpus.block_types() == ["section_header", "table", "text"]


def test_a_re_read_walks_every_passage_with_its_documents_language(corpus) -> None:
    """The document's and not the passage's: it is the fallback."""
    walked = corpus.texts()

    assert len(walked) == 3
    assert {language for _, _, language in walked} == {"de", "en"}


def test_a_re_read_can_be_narrowed_to_one_document(corpus) -> None:
    """This stage queues over documents, so that is what it narrows on."""
    from database.qa_generator import Passage

    walked = corpus.texts(Passage.doc_sha256 == B)

    assert [language for _, _, language in walked] == ["en"]


def test_a_re_read_replaces_the_language_and_the_lemmas(corpus, engine) -> None:
    """Both, so a change to either reaches the topic model."""
    with engine.connect() as connection:
        held = connection.execute(
            text(
                "SELECT id FROM passages WHERE ordinal = 1 AND doc_sha256 = :a",
            ),
            {"a": A},
        ).scalar_one()

    assert corpus.revocabulary([(held, "en", ["institution", "figure"])]) == 1

    row = one(engine, "SELECT language, lemmas FROM passages WHERE id = :id", id=held)
    assert row.language == "en"
    assert row.lemmas == ["institution", "figure"]


def test_a_re_read_leaves_the_sentence_offsets_alone(corpus, engine) -> None:
    """A fact cites one by index, so moving them moves every citation."""
    with engine.connect() as connection:
        held = connection.execute(
            text("SELECT id FROM passages WHERE ordinal = 1 AND doc_sha256 = :a"),
            {"a": A},
        ).scalar_one()
    before = one(engine, "SELECT sentences FROM passages WHERE id = :id", id=held)

    corpus.revocabulary([(held, "en", ["institution"])])

    after = one(engine, "SELECT sentences, text FROM passages WHERE id = :id", id=held)
    assert after.sentences == before.sentences
    assert after.text == "Die Institute melden ihre Kennzahlen."


def test_a_re_read_of_nothing_writes_nothing(corpus) -> None:
    """An empty batch is not an empty update statement."""
    assert corpus.revocabulary([]) == 0


def test_a_re_read_may_write_no_language_at_all(corpus, engine) -> None:
    """Which is what a passage too short to judge gets."""
    with engine.connect() as connection:
        held = connection.execute(text("SELECT min(id) FROM passages")).scalar_one()

    assert corpus.revocabulary([(held, None, [])]) == 1
    assert (
        one(engine, "SELECT language FROM passages WHERE id = :id", id=held).language
        is None
    )


def test_a_re_read_that_writes_an_empty_language_is_refused(corpus, engine) -> None:
    """The empty string is not a code, and the column says so."""
    with engine.connect() as connection:
        held = connection.execute(text("SELECT min(id) FROM passages")).scalar_one()

    with pytest.raises(IntegrityError, match="passages_language_is_iso_639_1"):
        corpus.revocabulary([(held, "", [])])


# ── The document catalogue ─────────────────────────────────────────────────


@pytest.fixture
def catalogue(engine, database):
    """Two documents, one uploaded twice under two names."""
    with Session(engine) as session:
        session.add_all(
            [
                document(A, title="Jahresbericht 2026", language="de"),
                document(B, title="Annual Report", language="en"),
            ]
        )
        session.flush()
        session.add_all(
            [
                event(submitted_filename="first-name.pdf", sha256=A),
                event(submitted_filename="second-name.pdf", sha256=A),
                event(submitted_filename="other.pdf", sha256=B),
                event(
                    submitted_filename="refused.pdf",
                    sha256=None,
                    outcome="too_large",
                    detail="too big",
                ),
            ]
        )
        session.commit()
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE ingest_events SET submitted_at = now() - interval '2 days' "
                "WHERE submitted_filename = 'first-name.pdf'"
            )
        )
        connection.execute(
            text(
                "UPDATE ingest_events SET submitted_at = now() - interval '1 day' "
                "WHERE submitted_filename = 'other.pdf'"
            )
        )
    return DocumentRepository()


def test_a_document_is_named_by_the_upload_it_first_arrived_under(catalogue) -> None:
    """The earliest event's name, not the alphabetically smallest."""
    _, listed = catalogue.page()

    assert {one.sha256: one.filename for one in listed}[A] == "first-name.pdf"


def test_the_name_and_the_date_come_from_the_same_upload(catalogue) -> None:
    """Two independent aggregates would take them from different rows."""
    _, listed = catalogue.page()

    first = next(one for one in listed if one.sha256 == A)
    assert first.filename == "first-name.pdf"
    assert first.first_seen is not None


def test_a_refused_upload_is_no_document(catalogue) -> None:
    """It writes an event and nothing else, so it cannot be listed."""
    total, listed = catalogue.page()

    assert total == 2
    assert {one.sha256 for one in listed} == {A, B}


def test_the_newest_upload_is_listed_first(catalogue) -> None:
    """Which is the order the Documents page shows."""
    _, listed = catalogue.page()

    assert [one.sha256 for one in listed] == [B, A]


def test_a_document_whose_upload_record_is_gone_sorts_last(catalogue, engine) -> None:
    """DESC puts NULLs first in PostgreSQL, above every real document."""
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM ingest_events WHERE sha256 = :b"), {"b": B}
        )

    _, listed = catalogue.page()

    assert [one.sha256 for one in listed] == [A, B]
    assert listed[-1].filename is None


def test_the_listing_counts_the_passages_and_the_ones_read_for_facts(
    catalogue, engine
) -> None:
    """Per document, not grouped over the whole corpus."""
    with Session(engine) as session:
        session.add_all(
            [
                passage(A, ordinal=1, extract_status=Status.EXTRACTED),
                passage(A, ordinal=2),
                passage(B, ordinal=1),
            ]
        )
        session.commit()

    _, listed = catalogue.page()

    held = {one.sha256: (one.total_passages, one.extracted_passages) for one in listed}
    assert held == {A: (2, 1), B: (1, 0)}


@pytest.mark.parametrize(
    ("search", "found"),
    [
        ("first-name", {"a"}),
        ("Jahresbericht", {"a"}),
        ("Annual", {"b"}),
        ("nothing at all", set()),
    ],
)
def test_the_search_looks_in_the_filename_the_title_and_the_digest(
    catalogue, search, found
) -> None:
    """A person looking for a document does not know which they remember."""
    total, listed = catalogue.page(search)

    assert total == len(found)
    assert {one.sha256 for one in listed} == {digest(letter) for letter in found}


def test_only_the_first_name_a_document_arrived_under_is_searchable(
    catalogue,
) -> None:
    """The listing carries one name per document, and it is the earliest.

    A re-upload under another name is recorded in ingest_events and does not
    become a second way to find the document.
    """
    assert catalogue.page("first-name")[0] == 1
    assert catalogue.page("second-name")[0] == 0


def test_a_document_is_found_by_its_own_digest(catalogue) -> None:
    """Which is what a log line or a URL carries."""
    total, listed = catalogue.page(A[:12])

    assert total == 1
    assert listed[0].sha256 == A


def test_the_count_and_the_page_agree_about_the_search(catalogue) -> None:
    """One filter, applied in one place."""
    total, listed = catalogue.page("pdf", limit=1)

    assert total == 2
    assert len(listed) == 1


def test_a_picker_is_offered_every_document_newest_first(catalogue) -> None:
    """The cheapest listing, because three pages poll it."""
    named = catalogue.names()

    assert [one.sha256 for one in named] == [B, A]
    assert [one.filename for one in named] == ["other.pdf", "first-name.pdf"]


def test_the_service_reports_what_it_owns(catalogue) -> None:
    """Documents and every attempt, refused ones included."""
    assert catalogue.counts() == {"documents": 2, "upload_attempts": 4}


def test_an_exact_digest_is_what_says_the_bytes_are_held(catalogue) -> None:
    """Duplicate detection is byte-exact."""
    assert catalogue.exists(A) is True
    assert catalogue.exists(digest("z")) is False


def test_the_media_type_is_read_back_for_the_key_it_builds(catalogue) -> None:
    """And a digest nothing holds answers nothing."""
    assert catalogue.media_type(A) == "application/pdf"
    assert catalogue.media_type(digest("z")) is None


def test_deleting_a_document_takes_its_passages_and_keeps_its_events(
    catalogue, engine
) -> None:
    """The audit trail outlives the document it describes."""
    with Session(engine) as session:
        session.add_all([passage(A, ordinal=1), passage(A, ordinal=2)])
        session.commit()

    assert catalogue.delete(A) == 2

    assert one(engine, "SELECT count(*) AS held FROM documents").held == 1
    assert one(engine, "SELECT count(*) AS held FROM passages").held == 0
    assert one(engine, "SELECT count(*) AS held FROM ingest_events").held == 4
    assert rows(engine, "SELECT sha256 FROM ingest_events WHERE sha256 IS NULL") != []


def test_dropping_the_derived_data_keeps_the_row_and_requeues_chunking(
    catalogue, engine
) -> None:
    """`new`, which is the state a freshly parsed document is in."""
    with Session(engine) as session:
        session.add(document(digest("c"), chunk_status=Status.CHUNKED))
        session.add(passage(digest("c"), ordinal=1))
        session.commit()

    assert catalogue.delete_derived(digest("c")) == 1

    row = one(
        engine,
        "SELECT chunk_status FROM documents WHERE sha256 = :sha",
        sha=digest("c"),
    )
    assert row.chunk_status == Status.NEW
    assert one(engine, "SELECT count(*) AS held FROM passages").held == 0


def test_deleting_a_document_that_is_not_there_removes_nothing(catalogue) -> None:
    """A count of zero rather than a failure."""
    assert catalogue.delete(digest("z")) == 0
    assert catalogue.delete_derived(digest("z")) == 0
