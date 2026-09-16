"""Each page, run against a scripted backend.

A page that raises renders a Streamlit traceback and nothing else, so the
thing worth asserting is that a refusal reaches the reader as a message.
"""

from __future__ import annotations

import pytest
import requests
from conftest import Answers

pytestmark = pytest.mark.frontend

#: What the backend answers when it is down.
UNREACHABLE = requests.exceptions.ConnectionError("connection refused")

HEALTHY = {
    "database": {"ok": True, "detail": "Connected, schema applied.", "metrics": {}},
    "object_store": {
        "ok": True,
        "detail": "documents, export, parsed",
        "metrics": {"documents": 2, "parsed": 1, "export": 0},
    },
    "ingestion": {"ok": True, "detail": "Documents held.", "metrics": {"documents": 2}},
}

EMPTY_PAGE = {
    "total": 0,
    "documents": [],
    "passages": [],
    "facts": [],
    "questions": [],
}

#: One accepted question, as /questions returns it.
QUESTION = {
    "id": 1,
    "question_text": "Within how many hours is a standard request answered?",
    "target_answer": "48 hours",
    "answerable": True,
    "difficulty": "hard",
    "passage_scope": "multi_passage",
    "document_scope": "cross_document",
    "topic_scope": "multi_topic",
    "answer_chars": 8,
    "thread_position": 1,
    "follows_id": None,
    "question_type": "comparison",
    "answer_form": "list",
    "planned_difficulty": "hard",
    "language": "en",
    "status": "accepted",
    "rejected_reason": None,
    "created_at": "2026-09-14T10:00:00",
    "facts": 2,
    "documents": ["a" * 64, "b" * 64],
    "topics": ["Support"],
}

#: What /questions/quality returns for that one question.
QUESTION_QUALITY = {
    "total": 1,
    "accepted": 1,
    "draft": 0,
    "unanswerable": 0,
    "topics_covered": 1,
    "topics_in_coverage": 2,
    "followups": 0,
    "mean_question_chars": 54.0,
    "mean_answer_chars": 8.0,
    "rejected": {},
    "difficulty": {"hard": 1},
    "passage_scope": {"multi_passage": 1},
    "document_scope": {"cross_document": 1},
    "topic_scope": {"multi_topic": 1},
    "question_type": {"comparison": 1},
    "answer_form": {"list": 1},
    "planned_difficulty": {"hard": 1},
    "planned_met": 1,
}

#: What /questions/plan returns: the mix that was asked for.
QUESTION_PLAN = {
    "types": {"factoid": 3, "comparison": 1},
    "difficulty": {"easy": 2, "hard": 1},
    "followup_types": ["condition", "reason"],
    "per_topic": 20,
    "unanswerable_share": 0.25,
    "followup_share": 0.3,
    "max_followups": 2,
    "answer_chars": {"value": [1, 80], "list": [3, 300], "explanation": [20, 600]},
}


def text_of(app) -> str:
    """Everything the page rendered, as one string."""
    parts = []
    for kind in (
        "markdown",
        "text",
        "info",
        "warning",
        "error",
        "caption",
        "title",
        "subheader",
    ):
        parts += [element.value for element in getattr(app, kind)]
    return " ".join(str(part) for part in parts)


def test_the_health_page_draws_the_figures_the_backend_reports(run_view) -> None:
    """The panel renders whatever is there without knowing the names.

    A component reporting no figures - the database - contributes a card
    and no metric, which is why this counts the figures rather than the
    components.
    """
    app = run_view(
        "health",
        health_api=Answers(reachable=(True, "ok"), components=HEALTHY),
    )

    assert not app.exception
    drawn = [element.value for element in app.metric]
    assert drawn == ["2", "1", "0", "2"], drawn


def test_the_health_page_names_each_group_of_figures(run_view) -> None:
    """One heading per component that has any."""
    app = run_view(
        "health",
        health_api=Answers(reachable=(True, "ok"), components=HEALTHY),
    )

    headings = [element.value for element in app.markdown]
    assert "##### Object Store" in headings, headings
    assert "##### Ingestion" in headings, headings


def test_the_health_page_survives_a_backend_that_is_down(run_view) -> None:
    """The one page that must render when nothing else can."""
    app = run_view(
        "health",
        health_api=Answers(
            reachable=(False, "connection refused"), components=UNREACHABLE
        ),
    )

    assert not app.exception, "the health page raised instead of reporting"


@pytest.mark.parametrize(
    ("view", "factory"),
    [
        ("documents", "catalog_api"),
        ("passages", "catalog_api"),
        ("facts", "catalog_api"),
        ("questions", "catalog_api"),
    ],
)
def test_a_listing_page_renders_when_the_corpus_is_empty(
    run_view, view, factory
) -> None:
    """The first thing a new deployment shows."""
    app = run_view(
        view,
        **{
            factory: Answers(
                documents=EMPTY_PAGE,
                passages=EMPTY_PAGE,
                facts=EMPTY_PAGE,
                fact_quality={},
                questions=EMPTY_PAGE,
                question_quality=QUESTION_QUALITY,
                question_plan=QUESTION_PLAN,
                document_names=[],
                passage_types=[],
                stage_status={"stage": view, "working": False, "rows": {}},
            )
        },
    )

    assert not app.exception, app.exception


@pytest.mark.parametrize(
    ("view", "factory"),
    [
        ("documents", "catalog_api"),
        ("passages", "catalog_api"),
        ("facts", "catalog_api"),
        ("topics", "catalog_api"),
        ("questions", "catalog_api"),
        ("upload", "upload_api"),
    ],
)
def test_a_page_reports_an_unreachable_backend_rather_than_raising(
    run_view, view, factory
) -> None:
    """Every call the page makes fails; the reader is told, not shown a stack."""
    app = run_view(
        view,
        **{
            factory: Answers(
                documents=UNREACHABLE,
                passages=UNREACHABLE,
                facts=UNREACHABLE,
                fact_quality=UNREACHABLE,
                questions=UNREACHABLE,
                question_quality=UNREACHABLE,
                question_plan=UNREACHABLE,
                document_names=UNREACHABLE,
                passage_types=UNREACHABLE,
                topics=UNREACHABLE,
                fit=UNREACHABLE,
                stage_status=UNREACHABLE,
                add_document=UNREACHABLE,
            )
        },
    )

    assert not app.exception, f"{view} raised: {app.exception}"


def test_the_documents_page_lists_what_the_backend_returns(run_view) -> None:
    """The listing is drawn from the answer, not from a second call."""
    held = {
        "sha256": "a" * 64,
        "filename": "annual-report.pdf",
        "title": "Annual Report",
        "page_count": 12,
        "language": "en",
        "parse_status": "parsed",
        "parse_error": None,
        "chunk_status": "chunked",
        "chunk_error": None,
        "extracted_passages": 3,
        "total_passages": 5,
        "first_seen": "2026-09-14T10:00:00",
        "oversized": 0,
    }
    app = run_view(
        "documents",
        catalog_api=Answers(
            documents={"total": 1, "documents": [held]},
            document_names=[{"sha256": "a" * 64, "filename": "annual-report.pdf"}],
            stage_status={"stage": "parsing", "working": False, "rows": {}},
            fact_quality={},
        ),
    )

    assert not app.exception, app.exception
    assert "annual-report.pdf" in text_of(app) or app.dataframe


def test_the_upload_page_offers_a_file_picker(run_view) -> None:
    """The one control that starts everything."""
    app = run_view("upload", upload_api=Answers(add_document={}))

    assert not app.exception
    assert app.get("file_uploader"), "no file picker on the upload page"


def test_the_questions_page_lists_what_the_backend_returns(run_view) -> None:
    """The listing and the gate table are drawn from one answer each."""
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={"total": 1, "questions": [QUESTION]},
            question_quality=QUESTION_QUALITY,
            question_plan=QUESTION_PLAN,
            question={"question": QUESTION, "sources": [], "thread": [QUESTION]},
            document_names=[],
            stage_status={
                "stage": "questions",
                "working": False,
                "rows": {"generated": 1, "new": 1},
            },
        ),
    )

    assert not app.exception, app.exception
    assert app.dataframe, "nothing was tabled"


def test_the_questions_page_says_when_a_cited_fact_no_longer_holds(run_view) -> None:
    """A question resting on a rejected fact is the quiet kind of wrong."""
    moved = {
        "fact_id": 9,
        "statement": "A standard request is answered within 48 hours.",
        "evidence_text": "Standard requests are answered within 48 hours.",
        "validated": False,
        "passage_id": 4,
        "doc_sha256": "a" * 64,
        "ordinal": 2,
    }
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={"total": 1, "questions": [QUESTION]},
            question_quality=QUESTION_QUALITY,
            question_plan=QUESTION_PLAN,
            question={"question": QUESTION, "sources": [moved], "thread": [QUESTION]},
            document_names=[],
            stage_status={"stage": "questions", "working": False, "rows": {}},
        ),
    )

    assert not app.exception, app.exception
    assert "questions-reverify" in text_of(app)


def test_accepting_a_question_writes_the_status_and_nothing_else(run_view) -> None:
    """The one control on this page that changes a row."""
    client = Answers(
        questions={
            "total": 1,
            "questions": [
                {**QUESTION, "status": "rejected", "rejected_reason": "duplicate"}
            ],
        },
        question_quality={
            **QUESTION_QUALITY,
            "accepted": 0,
            "rejected": {"duplicate": 1},
        },
        question_plan=QUESTION_PLAN,
        question={"question": {**QUESTION, "status": "rejected"}, "sources": []},
        decide_question={},
        document_names=[],
        stage_status={"stage": "questions", "working": False, "rows": {}},
    )
    app = run_view("questions", catalog_api=client)

    accepting = [one for one in app.button if one.label == "Accept"]
    assert accepting, "no accept control on the page"
    accepting[0].click().run()

    assert ("decide_question", (1, "accepted"), {}) in client.asked


def test_starting_generation_queues_the_topics(run_view) -> None:
    """The Start control on the Questions page, which acts on every topic.

    Generation queues over topics rather than documents, so this page has no
    per-item controls: the four verbs act on the whole corpus, and Start is
    the one that makes a fitted topic claimable.
    """
    client = Answers(
        questions={"total": 1, "questions": [QUESTION]},
        question_quality=QUESTION_QUALITY,
        question_plan=QUESTION_PLAN,
        question={"question": QUESTION, "sources": []},
        stage_action={"detail": "24 row(s) queued."},
        document_names=[],
        stage_status={"stage": "questions", "working": False, "rows": {"new": 24}},
    )
    app = run_view("questions", catalog_api=client)

    starting = [one for one in app.button if one.label == "Start"]
    assert starting, "no Start control on the page"
    starting[0].click().run()

    assert ("stage_action", ("questions", "start", None), {}) in client.asked


def test_the_start_control_is_dead_when_every_topic_is_already_queued(run_view) -> None:
    """A control that would do nothing is greyed rather than hidden."""
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={"total": 1, "questions": [QUESTION]},
            question_quality=QUESTION_QUALITY,
            question_plan=QUESTION_PLAN,
            question={"question": QUESTION, "sources": [], "thread": [QUESTION]},
            document_names=[],
            stage_status={
                "stage": "questions",
                "working": True,
                "rows": {"pending": 24},
            },
        ),
    )

    starting = [one for one in app.button if one.label == "Start"]
    assert starting and starting[0].disabled


def test_the_page_survives_a_rejection_code_it_has_never_heard_of(run_view) -> None:
    """A gate added in the backend must not blank the page that reports it."""
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={
                "total": 1,
                "questions": [
                    {**QUESTION, "status": "rejected", "rejected_reason": "invented"}
                ],
            },
            question_quality={**QUESTION_QUALITY, "rejected": {"invented": 1}},
            question_plan=QUESTION_PLAN,
            question={"question": QUESTION, "sources": [], "thread": [QUESTION]},
            document_names=[],
            stage_status={"stage": "questions", "working": False, "rows": {}},
        ),
    )

    assert not app.exception, app.exception
    # The gate table is a dataframe, so its rows are not in text_of().
    tabled = " ".join(str(one.value) for one in app.dataframe)
    assert "invented" in tabled, "the unknown gate was not listed"
    assert "no description" in tabled, "it was listed without saying what it is"


def test_a_question_with_no_difficulty_or_answer_still_renders(run_view) -> None:
    """Both columns are nullable, and a page that assumes otherwise breaks."""
    bare = {
        **QUESTION,
        "target_answer": None,
        "answerable": False,
        "difficulty": None,
        "passage_scope": None,
        "document_scope": None,
        "topic_scope": None,
        "answer_chars": None,
        "documents": [],
        "topics": [],
        "facts": 0,
    }
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={"total": 1, "questions": [bare]},
            question_quality={
                **QUESTION_QUALITY,
                "difficulty": {},
                "passage_scope": {},
                "document_scope": {},
                "topic_scope": {},
            },
            question={"question": bare, "sources": [], "thread": [bare]},
            document_names=[],
            stage_status={"stage": "questions", "working": False, "rows": {}},
        ),
    )

    assert not app.exception, app.exception


def test_the_questions_page_says_which_kinds_were_asked_for_and_written(
    run_view,
) -> None:
    """A mix nobody got must be visible as a mix nobody got.

    The plan decides what a run writes; the page is where somebody reads
    whether the corpus supported it.
    """
    app = run_view(
        "questions",
        catalog_api=Answers(
            questions={"total": 1, "questions": [QUESTION]},
            question_quality=QUESTION_QUALITY,
            question_plan=QUESTION_PLAN,
            question={"question": QUESTION, "sources": [], "thread": [QUESTION]},
            document_names=[],
            stage_status={"stage": "questions", "working": False, "rows": {}},
        ),
    )

    assert not app.exception, app.exception
    tabled = " ".join(str(one.value) for one in app.dataframe)
    assert "comparison" in tabled, "the kind written was not listed"
    assert "factoid" in tabled, "a kind asked for and never written was not listed"


#: One passage, as /passages returns it in a listing.
PASSAGE = {
    "id": 11,
    "doc_sha256": "a" * 64,
    "ordinal": 3,
    "text": "| Modell | Masse |\n| Kompakt | 4 |",
    "page_from": 4,
    "page_to": 5,
    "section_path": "2 Tabellen",
    "block_type": "table",
    "language": "de",
    "doc_item_refs": ["#/tables/0"],
    "bbox": [{"page": 4, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}],
    "table_count": 1,
    "sentence_count": 2,
}

#: That passage in full, as /passages/{id} returns it.
PASSAGE_DETAIL = {
    "passage": PASSAGE,
    "table_cells": [
        {
            "caption": "Tabelle 1",
            "num_rows": 2,
            "num_cols": 2,
            "cells": [
                {
                    "row": 0,
                    "col": 0,
                    "row_span": 1,
                    "col_span": 1,
                    "column_header": True,
                    "row_header": False,
                    "text": "Modell",
                    "line": None,
                },
                {
                    "row": 1,
                    "col": 0,
                    "row_span": 1,
                    "col_span": 1,
                    "column_header": False,
                    "row_header": True,
                    "text": "Kompakt",
                    "line": 1,
                },
            ],
        }
    ],
    "sentences": [
        {"i": 0, "start": 0, "end": 18, "predicates": 0},
        {"i": 1, "start": 19, "end": 34, "predicates": 0},
    ],
    "extract_status": "extracted",
    "extract_error": None,
}


def passages_backend(**overrides) -> Answers:
    """A backend that answers every call the Passages page makes."""
    return Answers(
        **{
            "passages": {"total": 1, "passages": [PASSAGE]},
            "passage": PASSAGE_DETAIL,
            "document_names": [{"sha256": "a" * 64, "filename": "report.pdf"}],
            "passage_types": ["table", "text"],
            "stage_status": {"working": False, "rows": {"chunked": 1, "extracted": 1}},
            **overrides,
        }
    )


def test_the_passages_page_lists_what_the_backend_returns(run_view) -> None:
    """One row per passage, drawn from the answer and not a second call."""
    app = run_view("passages", catalog_api=passages_backend())

    assert not app.exception, app.exception
    listed = app.dataframe[0].value
    assert listed["#"].tolist() == [3]
    assert listed["Pages"].tolist() == ["4–5"]
    assert listed["Language"].tolist() == ["DE"]


def test_the_passages_page_survives_a_page_whose_rows_have_gone(run_view) -> None:
    """The count and the rows are two queries, so they can disagree.

    A delete landing between them leaves a total with no rows, and the
    lengths measured on those rows have nothing to measure.
    """
    app = run_view(
        "passages", catalog_api=passages_backend(passages={"total": 3, "passages": []})
    )

    assert not app.exception, f"the page raised: {app.exception}"
    assert "Failed to load" not in text_of(app), text_of(app)


def test_the_passages_page_numbers_the_sentences_a_fact_would_cite(run_view) -> None:
    """The number in the first column is what a citation refers to."""
    app = run_view("passages", catalog_api=passages_backend())

    numbered = next(
        frame.value for frame in app.dataframe if "Claims" in frame.value.columns
    )
    assert numbered["#"].tolist() == [0, 1]
    assert numbered["Text"].tolist()[0] == "| Modell | Masse |"


def test_the_passages_page_shows_which_row_each_cell_cites(run_view) -> None:
    """A header cell cites nothing; a data cell cites the row it sits in."""
    app = run_view("passages", catalog_api=passages_backend())

    cells = next(
        frame.value for frame in app.dataframe if "Cites row" in frame.value.columns
    )
    assert cells["Cites row"].tolist() == ["—", "1"]
    assert cells["Header"].tolist() == ["column", "row"]


def test_the_passages_page_reports_a_passage_extraction_could_not_read(
    run_view,
) -> None:
    """The reason, against the passage it belongs to."""
    app = run_view(
        "passages",
        catalog_api=passages_backend(
            passage={
                **PASSAGE_DETAIL,
                "extract_status": "failed",
                "extract_error": "the model did not answer",
            }
        ),
    )

    assert not app.exception
    assert "the model did not answer" in text_of(app)


def test_the_documents_page_reports_a_stage_that_failed(run_view) -> None:
    """Parsing and chunking both record their failure on the document row."""
    app = run_view(
        "documents",
        catalog_api=Answers(
            documents={
                "total": 1,
                "documents": [
                    {
                        "sha256": "a" * 64,
                        "filename": "report.pdf",
                        "first_seen": "2026-09-14T10:00:00",
                        "page_count": 12,
                        "title": None,
                        "language": None,
                        "parse_status": "failed",
                        "parse_error": "the converter puts this at 0.31",
                        "chunk_status": "new",
                        "chunk_error": "a failure from an earlier run",
                        "extracted_passages": 0,
                        "total_passages": 0,
                        "oversized": 0,
                    }
                ],
            },
            stage_status={"working": False, "rows": {}},
        ),
    )

    assert not app.exception, app.exception
    drawn = text_of(app)
    assert "the converter puts this at 0.31" in drawn
    assert "a failure from an earlier run" not in drawn, (
        "an error from a stage that is not failed was shown as its state"
    )
