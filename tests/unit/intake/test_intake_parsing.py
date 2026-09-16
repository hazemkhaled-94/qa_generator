"""Dispatching, converting, judging and recording one document.

The converter itself is stood in for: its layout and table models are a
download. What it produces is a real DoclingDocument, so everything read out
of one here is read the way parsing reads it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from docling_core.types.doc.labels import DocItemLabel
from intake_drivers import (
    ParseDriver,
    ScriptedPipeline,
    docling_document,
)

from preprocessing.parsing.analysis import (
    DocumentAnalyser,
    EmptyDocument,
)
from preprocessing.parsing.models import ClaimedDocument, Conversion, SourceDocument
from preprocessing.parsing.pipelines import (
    ConversionFailed,
    PdfPipeline,
    Pipeline,
    PipelineRegistry,
    UnsupportedFormat,
)
from preprocessing.parsing.pipelines.pdf import _score

SHA = "a" * 64
OTHER = "b" * 64


def claimed(
    sha256: str = SHA,
    media_type: str = "application/pdf",
    pages: int | None = 10,
    chars: int | None = 50_000,
) -> ClaimedDocument:
    """One document as the queue hands it over."""
    return ClaimedDocument(
        sha256=sha256, media_type=media_type, page_count=pages, char_count=chars
    )


# ── The registry ───────────────────────────────────────────────────────────


class _Other(Pipeline):
    """A second pipeline, for the tests about dispatch."""

    media_types = ("application/pdf",)

    def convert(self, source: SourceDocument) -> Conversion:
        """Never called."""
        raise NotImplementedError


def test_two_pipelines_claiming_one_type_is_refused() -> None:
    """Dispatch would be whichever was registered last."""
    with pytest.raises(ValueError, match="two pipelines claim application/pdf"):
        PipelineRegistry((ScriptedPipeline(), _Other()))


def test_a_type_no_pipeline_handles_names_the_ones_that_are() -> None:
    """Which is what says the allowlist has drifted."""
    registry = PipelineRegistry((ScriptedPipeline(),))

    with pytest.raises(UnsupportedFormat, match="known: application/pdf"):
        registry.for_media_type("image/tiff")


def test_the_registry_publishes_what_it_can_dispatch() -> None:
    """Sorted, so the message above reads the same every time."""
    assert PipelineRegistry((ScriptedPipeline(),)).media_types == ("application/pdf",)


# ── The scanned decision ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("pages", "chars", "scanned"),
    [
        (10, None, True),
        (None, 50_000, True),
        (0, 50_000, True),
        (10, 0, True),
        (10, 999, True),
        (10, 1000, False),
        (10, 50_000, False),
        (1, 99, True),
        (1, 100, False),
    ],
)
def test_a_document_is_judged_scanned_from_the_counts_ingestion_took(
    pages, chars, scanned
) -> None:
    """Characters per page against the threshold, so it costs nothing."""
    driver = ParseDriver(queued=(claimed(pages=pages, chars=chars),))

    driver.run()

    assert driver.scanned_as() is scanned


def test_a_scanned_pdf_is_refused_before_the_converter_is_built() -> None:
    """The reason is still known here; three steps on it is "no body text"."""
    pipeline = PdfPipeline(
        table_mode="fast",
        heading_hierarchy=False,
        document_timeout_seconds=None,
        artifacts_path=None,
    )

    with pytest.raises(ConversionFailed, match="without OCR"):
        pipeline.convert(SourceDocument(sha256=SHA, data=b"%PDF-", scanned=True))


# ── The flow ───────────────────────────────────────────────────────────────


def test_an_empty_queue_converts_nothing() -> None:
    """Draining an empty queue is not an error."""
    driver = ParseDriver()

    assert driver.run() is None
    assert driver.pipeline.calls == 0


def test_the_file_is_read_under_the_media_type_recorded_at_ingest() -> None:
    """Which is what makes the key content-addressed rather than guessed."""
    driver = ParseDriver(queued=(claimed(),))

    driver.run()

    assert driver.pipeline.converted[0].sha256 == SHA
    assert (
        driver.pipeline.converted[0].data
        == driver.documents.objects[driver.documents.key_for(SHA, "application/pdf")]
    )


def test_the_converted_document_is_stored_before_the_row_is_written() -> None:
    """A bug in the analysis must not mean paying for the conversion twice."""
    driver = ParseDriver(queued=(claimed(),))

    driver.run()

    key = driver.parsed.key_for(SHA)
    assert driver.parsed_keys == [key]
    assert driver.parsed.content_types[key] == "application/json"
    assert driver.parsed.metadata[key] == {"sha256": SHA}
    assert json.loads(driver.parsed.objects[key])["name"] == "t"


def test_what_the_analyser_read_is_what_the_row_records() -> None:
    """Title, language, content digest and both confidences."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(
            document=docling_document(
                title="Annual Report",
                paragraphs=(
                    "The supervisory authority publishes its findings every year.",
                ),
                pages=((1, 200.0, 300.0),),
            ),
            confidence=0.88,
            confidence_low=0.71,
        ),
    )

    assert driver.run() == SHA

    recorded = driver.completed[SHA]
    assert recorded.title == "Annual Report"
    assert recorded.language == "en"
    assert len(recorded.content_sha256) == 64
    assert (recorded.confidence, recorded.confidence_low) == (0.88, 0.71)
    assert recorded.page_count == 1


def test_a_document_no_pipeline_handles_is_failed_not_raised() -> None:
    """One unreadable file must not stop a batch."""
    driver = ParseDriver(queued=(claimed(media_type="image/tiff"),))

    assert driver.run() == SHA
    assert "no pipeline handles image/tiff" in (driver.failure or "")
    assert driver.completed == {}
    assert driver.parsed_keys == [], "nothing was converted, so nothing was stored"


def test_a_conversion_that_failed_is_recorded_with_its_reason() -> None:
    """The converter's own words, which name the file's problem."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(raises=ConversionFailed("the page tree is cyclic")),
    )

    driver.run()

    assert driver.failure == "the page tree is cyclic"
    assert driver.completed == {}


def test_a_conversion_with_no_body_text_is_failed() -> None:
    """A document converted into nothing is not a parsed document."""
    driver = ParseDriver(
        queued=(claimed(),), pipeline=ScriptedPipeline(document=docling_document())
    )

    driver.run()

    assert "no extractable body text" in (driver.failure or "")


def test_a_conversion_the_converter_is_unsure_of_is_failed() -> None:
    """Judged on the lower bound, and the failure carries both figures."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(confidence=0.9, confidence_low=0.31),
        min_confidence=0.5,
    )

    driver.run()

    assert "0.31" in (driver.failure or "")
    assert "0.50 floor" in (driver.failure or "")
    assert "PARSING_MIN_CONFIDENCE" in (driver.failure or "")


def test_a_conversion_exactly_at_the_floor_is_kept() -> None:
    """The floor is what is allowed, not what is refused."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(confidence_low=0.5),
        min_confidence=0.5,
    )

    driver.run()

    assert driver.failure is None
    assert SHA in driver.completed


def test_a_conversion_that_measured_no_confidence_is_kept() -> None:
    """An unmeasured score is NULL, and NULL is not below a floor."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(confidence=None, confidence_low=None),
        min_confidence=0.9,
    )

    driver.run()

    assert SHA in driver.completed
    assert driver.completed[SHA].confidence is None


def test_a_document_whose_text_the_corpus_already_holds_is_failed() -> None:
    """Different bytes, the same text: knowable only once the text exists."""
    analysed = DocumentAnalyser().analyse(
        Conversion(
            document=ScriptedPipeline().document, confidence=None, confidence_low=None
        )
    )
    driver = ParseDriver(queued=(claimed(),), holders={analysed.content_sha256: OTHER})

    driver.run()

    assert f"document {OTHER[:12]} already holds" in (driver.failure or "")
    assert driver.completed == {}


def test_a_document_holding_its_own_text_again_is_not_refused() -> None:
    """Re-parsing one document must not read as a duplicate of itself."""
    analysed = DocumentAnalyser().analyse(
        Conversion(
            document=ScriptedPipeline().document, confidence=None, confidence_low=None
        )
    )
    driver = ParseDriver(queued=(claimed(),), holders={analysed.content_sha256: SHA})

    driver.run()

    assert driver.failure is None
    assert SHA in driver.completed


def test_an_unexpected_failure_is_recorded_as_its_own_type() -> None:
    """Not swallowed, and not raised out of the drain loop either."""
    driver = ParseDriver(
        queued=(claimed(),), pipeline=ScriptedPipeline(raises=TypeError("not a stream"))
    )

    assert driver.run() == SHA
    assert driver.failure == "TypeError: not a stream"


def test_a_file_the_store_does_not_hold_is_failed() -> None:
    """A row pointing at a missing object fails this stage and only this one."""
    driver = ParseDriver(queued=(claimed(),))
    driver.documents.objects.clear()

    driver.run()

    assert "KeyError" in (driver.failure or "")


def test_a_failed_parse_leaves_its_converted_form_in_the_bucket() -> None:
    """Stored before the checks, and collected when the document is deleted."""
    driver = ParseDriver(
        queued=(claimed(),),
        pipeline=ScriptedPipeline(confidence_low=0.1),
        min_confidence=0.5,
    )

    driver.run()

    assert driver.failure is not None
    assert driver.parsed_keys == [driver.parsed.key_for(SHA)]


def test_every_queued_document_is_worked_and_one_failure_stops_nothing() -> None:
    """Which is what makes a batch a batch."""
    driver = ParseDriver(
        queued=(claimed(SHA), claimed(OTHER, media_type="image/tiff")),
    )

    assert driver.drain() == 2
    assert list(driver.completed) == [SHA]
    assert list(driver.failures) == [OTHER]


def test_a_drain_sweeps_the_claims_an_earlier_run_left_behind(caplog) -> None:
    """Before it claims anything of its own."""
    driver = ParseDriver(abandoned=3)

    with caplog.at_level("WARNING"):
        assert driver.drain() == 0

    assert "3 document(s) left claimed" in caplog.text


# ── The analyser ───────────────────────────────────────────────────────────


def analysed(document) -> object:
    """Reads one converted document the way the service reads it."""
    return DocumentAnalyser().analyse(
        Conversion(document=document, confidence=None, confidence_low=None)
    )


def test_the_title_is_the_documents_own_title_item() -> None:
    """Preferred over a section header, wherever each one sits."""
    document = docling_document(headings=("1 Scope",), title="Annual Report")
    document.add_text(label=DocItemLabel.TEXT, text="The authority publishes it.")

    assert analysed(document).title == "Annual Report"


def test_a_section_header_is_the_title_when_there_is_no_title_item() -> None:
    """A document whose converter found no title still has a name."""
    document = docling_document(
        headings=("1 Scope",), paragraphs=("The authority publishes it.",)
    )

    assert analysed(document).title == "1 Scope"


def test_a_title_of_only_whitespace_is_not_a_title() -> None:
    """A blank heading is not what a document is called."""
    document = docling_document(
        title="   ", headings=("1 Scope",), paragraphs=("Text.",)
    )

    assert analysed(document).title == "1 Scope"


def test_a_document_with_no_heading_at_all_has_no_title() -> None:
    """None, not the filename and not the first paragraph."""
    assert analysed(docling_document(paragraphs=("Text.",))).title is None


def test_a_document_with_no_body_text_is_refused() -> None:
    """A conversion that succeeded and produced nothing is not a document."""
    with pytest.raises(EmptyDocument, match="no extractable body text"):
        analysed(docling_document())


def test_the_content_hash_ignores_case_and_spacing() -> None:
    """The same text laid out differently is the same content."""
    spaced = analysed(docling_document(paragraphs=("The  Device   weighs 4 kg.",)))
    plain = analysed(docling_document(paragraphs=("the device weighs 4 kg.",)))

    assert spaced.content_sha256 == plain.content_sha256


def test_the_content_hash_folds_the_german_sharp_s() -> None:
    """A document written STRASSE and one written Straße are one document."""
    upper = analysed(docling_document(paragraphs=("DIE STRASSE IST GESPERRT.",)))
    mixed = analysed(docling_document(paragraphs=("Die Straße ist gesperrt.",)))

    assert upper.content_sha256 == mixed.content_sha256


def test_the_content_hash_keeps_digits_and_punctuation() -> None:
    """In financial text they are the content."""
    comma = analysed(
        docling_document(paragraphs=("Der Anteil liegt bei 12,5 Prozent.",))
    )
    point = analysed(
        docling_document(paragraphs=("Der Anteil liegt bei 12.5 Prozent.",))
    )

    assert comma.content_sha256 != point.content_sha256


def test_the_page_count_is_the_pages_the_converted_document_holds() -> None:
    """Ingestion's count stays ingestion's; this is the converter's."""
    document = docling_document(
        paragraphs=("Text.",), pages=((1, 200.0, 300.0), (2, 200.0, 300.0))
    )

    assert analysed(document).page_count == 2


# ── Confidence ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(0.84, 0.84), (0.0, 0.0), (1.0, 1.0), (float("nan"), None), (None, None)],
)
def test_a_score_reaches_the_database_as_a_value_it_accepts(reported, expected) -> None:
    """NaN compares equal to nothing, so a later filter would skip the row."""
    result = SimpleNamespace(confidence=SimpleNamespace(mean_score=reported))

    assert _score(result, "mean_score") == expected


def test_a_conversion_reporting_no_confidence_at_all_reads_as_none() -> None:
    """A converter that measures nothing is not a converter that scored zero."""
    assert _score(SimpleNamespace(), "mean_score") is None


def test_the_same_text_under_different_bytes_is_refused_once_it_is_read() -> None:
    """Ingestion catches a re-upload by its bytes; this catches a re-export.

    Two documents the corpus accepted separately, because their bytes differ,
    converting to the same text. The first keeps it and the second is refused
    by name.
    """
    driver = ParseDriver(queued=(claimed(SHA), claimed(OTHER)))

    assert driver.drain() == 2

    assert list(driver.completed) == [SHA]
    assert f"document {SHA[:12]} already holds" in driver.failures[OTHER]


def test_a_duplicate_by_content_keeps_its_file_and_its_converted_form() -> None:
    """Refused after both were written, so somebody has to delete the row."""
    driver = ParseDriver(queued=(claimed(SHA), claimed(OTHER)))

    driver.drain()

    assert sorted(driver.parsed_keys) == sorted(
        [driver.parsed.key_for(SHA), driver.parsed.key_for(OTHER)]
    )
    assert OTHER in driver.failures
