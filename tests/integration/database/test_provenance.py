"""Every artefact records the run and the trace that produced it.

One test per stage, against a real Postgres, because what is under test is
the write: each stage reads the ambient span where it stores its rows, and
a stage that stopped doing so would still pass its own tests.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from seed import digest, document, passage
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.qa_generator import Status
from telemetry.evaluations import current_ids

pytestmark = pytest.mark.integration

#: A tracer that mints real span ids and exports nothing. Without one every
#: span is the no-op, its context is invalid, and every assertion below
#: passes on two empty strings.
_TRACER = TracerProvider().get_tracer(__name__)


def _rows(engine, sql: str) -> list[tuple]:
    """Reads provenance columns back."""
    with engine.connect() as connection:
        return connection.execute(text(sql)).all()


def test_parsing_records_the_run_and_the_trace(engine, database) -> None:
    """A parsed document carries the span the parse ran in."""
    from preprocessing.parsing.models import ParsedDocument
    from preprocessing.parsing.repository import ParseQueue

    sha = digest("a")
    with Session(engine) as session:
        session.add(document(sha, parse_status=Status.PENDING))
        session.commit()

    queue = ParseQueue()
    assert queue.claim() is not None
    with _TRACER.start_as_current_span("parse"):
        trace_id, span_id = current_ids()
        queue.complete(
            sha,
            ParsedDocument(
                title="A report",
                language="en",
                content_sha256=digest("body"),
                page_count=1,
                confidence=0.9,
                confidence_low=0.8,
            ),
        )

    assert _rows(
        engine,
        "SELECT parse_trace_id, parse_span_id, parse_run_id IS NOT NULL "
        "FROM documents",
    ) == [(trace_id, span_id, True)]


def test_chunking_records_the_run_and_the_trace(engine, database) -> None:
    """Every passage of one document carries that document's chunk span."""
    from preprocessing.chunking.models import Chunk, Chunking
    from preprocessing.chunking.repository import ChunkQueue

    sha = digest("a")
    with Session(engine) as session:
        session.add(document(sha, chunk_status=Status.IN_PROGRESS))
        session.commit()

    chunks = [
        Chunk(
            ordinal=ordinal,
            text=f"Passage {ordinal}.",
            page_from=1,
            page_to=1,
            section_path=None,
            block_type="text",
            language="en",
        )
        for ordinal in (1, 2)
    ]
    with _TRACER.start_as_current_span("chunk"):
        trace_id, span_id = current_ids()
        ChunkQueue().replace(sha, Chunking(passages=chunks, oversized=0))

    assert _rows(
        engine,
        "SELECT trace_id, span_id, run_id IS NOT NULL FROM passages ORDER BY ordinal",
    ) == [(trace_id, span_id, True)] * 2


def test_extraction_records_the_trace(engine, database) -> None:
    """Every fact from one passage carries that passage's extract span."""
    from facts import FactStore, checked

    store = FactStore(engine)
    corpus = store.corpus(("a", ["Standard requests are answered within 48 hours."]))
    passage_id = corpus["a"][0]

    with _TRACER.start_as_current_span("extract"):
        trace_id, span_id = current_ids()
        store.store(passage_id, checked(passage_id))

    assert store.rows("trace_id", "span_id") == [(trace_id, span_id)]


def test_topics_record_the_span_of_their_own_language(engine, database) -> None:
    """A fit is one span per language, and each topic carries its own."""
    from dataclasses import replace as replaced

    from topics import TopicStore, fitting

    store = TopicStore(engine).given_passages(en=2, de=2)
    store.request()

    english = replaced(fitting("en"), trace_id="a" * 32, span_id="b" * 16)
    german = replaced(fitting("de"), trace_id="a" * 32, span_id="c" * 16)
    store.store(english, german)

    assert _rows(
        engine,
        "SELECT DISTINCT language, trace_id, span_id FROM topics ORDER BY language",
    ) == [("de", "a" * 32, "c" * 16), ("en", "a" * 32, "b" * 16)]
