"""One document, all the way through, in this process.

Every stage is the real service against the real database and the real
object store. Two things are stood in for, both because they reach outside
the machine: the PDF converter, whose layout models are a download, and the
served model, which is a GPU somewhere.

What this covers that the per-stage tests do not is the handover: each
stage leaves a row in the state the next one claims, and nothing starts
until something asks it to.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from seed import digest
from sqlalchemy import text

from database.qa_generator import Status
from extraction.extractors import ExtractorRegistry
from extraction.extractors.llm import LlmExtractor
from extraction.extractors.table import TableExtractor
from extraction.repository import PassageQueue
from extraction.service import ExtractionService
from extraction.validation import FactChecker
from ingestion.models import UploadedFile
from ingestion.repository import DocumentRepository
from ingestion.service import IngestService
from preprocessing.chunking.repository import ChunkQueue
from preprocessing.chunking.service import ChunkingService
from preprocessing.parsing.analysis import DocumentAnalyser
from preprocessing.parsing.models import Conversion, SourceDocument
from preprocessing.parsing.pipelines import Pipeline, PipelineRegistry
from preprocessing.parsing.repository import ParseQueue
from preprocessing.parsing.service import ParsingService

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

#: What the fake converter produces: two English paragraphs carrying claims
#: a deterministic reader will not take, so the model is what reads them.
PARAGRAPHS = (
    "The device weighs 4 kg and runs for 12 hours.",
    "The device arrives in March 2026 from the Hamburg plant.",
)

PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)


class FakePdfPipeline(Pipeline):
    """A converter that answers with a document instead of reading one.

    Docling's layout and table models are a download and a GPU's worth of
    work; what parsing does with a converted document is tested against
    real Docling documents in the unit layer.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/pdf",)

    def convert(self, source: SourceDocument) -> Conversion:
        """Builds a two-paragraph document."""
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        document = DoclingDocument(name=source.sha256[:12])
        document.add_title(text="The Device")
        for paragraph in PARAGRAPHS:
            document.add_text(label=DocItemLabel.TEXT, text=paragraph)
        return Conversion(document=document, confidence=0.9, confidence_low=None)


class StubModel:
    """A served model that reads one claim out of each sentence."""

    model = "ollama/stub"
    temperature = 0.0

    def __init__(self) -> None:
        """Initialises the call counter."""
        self.calls = 0

    def answer(self, *, system: str, user: str, shape):
        """Answers with one fact per numbered sentence in the excerpt."""
        self.calls += 1
        numbered = [line for line in user.splitlines() if line.startswith("[")]
        return shape(
            facts=[
                {
                    "sentences": [index],
                    "statement": "The device weighs 4 kg."
                    if "4 kg" in line
                    else "The device arrives in March 2026.",
                }
                for index, line in enumerate(numbered)
            ]
        )


@pytest.fixture(scope="module")
def chunker():
    """The real passage builder, or a skip when its tokenizer is not here.

    It is a HuggingFace tokenizer, downloaded on first use. A machine with
    no cache and no network cannot build one, and that is a reason to skip
    this rather than to stand it in for.
    """
    from preprocessing.chunking.passages import PassageBuilder

    try:
        return PassageBuilder(
            embedding_model="intfloat/multilingual-e5-large",
            max_tokens=512,
            merge_peers=True,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        pytest.skip(f"the chunker's tokenizer is unavailable: {exc}")


@pytest.fixture
def pipeline(database, buckets, chunker):
    """Every stage, wired as its factory wires it."""
    return _pipeline_with(chunker, StubModel())


def _pipeline_with(chunker, model) -> dict:
    """Builds the stages around one served model."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket

    return {
        "ingest": IngestService(
            repository=DocumentRepository(),
            store=DocumentsBucket(),
            max_file_size_bytes=1024 * 1024,
            allowed_media_types=("application/pdf",),
            pipeline_version="0.0.1",
        ),
        "parse": ParsingService(
            repository=ParseQueue(),
            documents=DocumentsBucket(),
            parsed=ParsedBucket(),
            pipelines=PipelineRegistry((FakePdfPipeline(),)),
            analyser=DocumentAnalyser(),
            ocr_char_threshold=100,
            min_confidence=0.5,
        ),
        "chunk": ChunkingService(
            repository=ChunkQueue(), parsed=ParsedBucket(), builder=chunker
        ),
        "extract": ExtractionService(
            repository=PassageQueue(),
            extractors=ExtractorRegistry(
                extractors=(TableExtractor(),), default=LlmExtractor(model)
            ),
            checker=FactChecker(),
        ),
        "model": model,
    }


def counts(engine) -> dict[str, int]:
    """How many rows each table holds."""
    with engine.connect() as connection:
        return {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar_one()
            for table in ("documents", "passages", "facts")
        }


def statuses(engine) -> tuple[str, str]:
    """The parse and chunk status of the one document."""
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT parse_status, chunk_status FROM documents")
        ).one()


def test_nothing_runs_until_something_asks_it_to(pipeline, engine) -> None:
    """A row arrives `new` and no worker looks at it."""
    pipeline["ingest"].ingest(UploadedFile("report.pdf", PDF))

    assert pipeline["parse"].drain() == 0
    assert statuses(engine) == (Status.NEW, Status.NEW)


def test_a_document_becomes_facts(pipeline, engine) -> None:
    """Upload, parse, chunk, extract, each stage claiming from the last."""
    stored = pipeline["ingest"].ingest(UploadedFile("report.pdf", PDF))
    assert stored.sha256

    ParseQueue().start()
    assert pipeline["parse"].drain() == 1
    assert statuses(engine)[0] == Status.PARSED

    ChunkQueue().start()
    assert pipeline["chunk"].drain() == 1
    assert statuses(engine)[1] == Status.CHUNKED

    PassageQueue().start()
    read = pipeline["extract"].drain()

    assert read > 0
    held = counts(engine)
    assert held["documents"] == 1
    assert held["passages"] > 0
    assert held["facts"] > 0


def test_every_stored_fact_cites_a_sentence_of_its_passage(pipeline, engine) -> None:
    """The citation is an index, so it has to index something."""
    _run(pipeline)

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT f.evidence_text, p.text, f.evidence_start, f.evidence_end "
                "FROM facts f JOIN passages p ON p.id = f.passage_id"
            )
        ).all()

    assert rows
    for evidence, passage, start, end in rows:
        assert passage[start:end] == evidence


def test_the_model_is_asked_once_per_passage_that_carries_a_claim(
    pipeline, engine
) -> None:
    """A heading or a caption never reaches it."""
    _run(pipeline)

    with engine.connect() as connection:
        passages = connection.execute(
            text("SELECT count(*) FROM passages")
        ).scalar_one()

    assert 0 < pipeline["model"].calls <= passages


def test_a_second_run_of_a_finished_stage_does_nothing(pipeline) -> None:
    """Draining an empty queue is not an error."""
    _run(pipeline)

    assert pipeline["parse"].drain() == 0
    assert pipeline["chunk"].drain() == 0
    assert pipeline["extract"].drain() == 0


def test_dropping_the_derived_data_lets_the_run_repeat(pipeline, engine) -> None:
    """Chunking returns to the queue without re-parsing."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket
    from ingestion.removal import RemovalService

    sha = _run(pipeline)
    before = counts(engine)["passages"]

    RemovalService(
        repository=DocumentRepository(),
        store=DocumentsBucket(),
        parsed=ParsedBucket(),
    ).delete_derived(sha)
    assert counts(engine)["passages"] == 0

    ChunkQueue().start()
    assert pipeline["chunk"].drain() == 1

    assert counts(engine)["passages"] == before
    assert statuses(engine)[0] == Status.PARSED, "it did not need re-parsing"


def test_a_model_that_will_not_answer_fails_the_passage(
    database, buckets, chunker, engine
) -> None:
    """The row records the reason and can be returned to the queue."""

    class Refusing:
        """A served model that is never available."""

        model, temperature = "ollama/stub", 0.0

        def answer(self, **_: object):
            """Refuses every ask."""
            from llm.client import ModelUnavailable

            raise ModelUnavailable("Timeout: the model did not answer")

    built = _pipeline_with(chunker, Refusing())
    _run(built)

    with engine.connect() as connection:
        failed = connection.execute(
            text(
                "SELECT extract_status, extract_error FROM passages "
                "WHERE extract_status = 'failed'"
            )
        ).all()

    assert failed, "no passage was marked failed"
    assert all("did not answer" in error for _, error in failed)
    assert PassageQueue().retry() == len(failed), "a failure must be retryable"


def _run(pipeline) -> str:
    """Runs one document through every stage, and answers with its digest."""
    stored = pipeline["ingest"].ingest(UploadedFile("report.pdf", PDF))
    ParseQueue().start()
    pipeline["parse"].drain()
    ChunkQueue().start()
    pipeline["chunk"].drain()
    PassageQueue().start()
    pipeline["extract"].drain()
    assert stored.sha256
    return stored.sha256


def test_the_digest_survives_every_stage(pipeline, engine) -> None:
    """One identity from the upload to the fact."""
    sha = _run(pipeline)

    with engine.connect() as connection:
        held = (
            connection.execute(
                text(
                    "SELECT DISTINCT p.doc_sha256 FROM passages p "
                    "JOIN facts f ON f.passage_id = p.id"
                )
            )
            .scalars()
            .all()
        )

    assert held == [sha]
    assert sha != digest("a")
