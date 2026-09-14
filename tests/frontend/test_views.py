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

EMPTY_PAGE = {"total": 0, "documents": [], "passages": [], "facts": []}


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
