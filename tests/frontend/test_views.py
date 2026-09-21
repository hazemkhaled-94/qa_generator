"""The rules every page keeps, checked on every page.

The pages differ in what they list and which stage they run. Everything else
about them is deliberately the same, and that sameness is what these cover:
one service per page, one row of figures, nothing below the table until a
row is picked, and no page reaching into another page's stage.
"""

from __future__ import annotations

import pytest
import requests
from conftest import Answers
from pages import (
    LISTING,
    OWNED,
    QUESTION,
    QUESTION_SOURCE,
    SHA,
    View,
    answers,
    changed,
    settings,
    status,
)

pytestmark = pytest.mark.frontend

#: Every page, including the two that list nothing.
EVERY = (*LISTING, "upload", "health")

#: A question the gates have not judged, so Accept is live on it.
QUESTION_DRAFT = changed(QUESTION, status="draft", rejected_reason=None)

#: What the two pages outside the catalogue are stubbed with.
OUTSIDE = {
    "upload": {"upload_api": Answers(counts={"documents": 3, "upload_attempts": 5})},
    "health": {
        "health_api": Answers(
            reachable=(True, "Reachable."),
            services=[
                {
                    "name": "argilla",
                    "purpose": "Where a person accepts or rejects.",
                    "ok": True,
                    "detail": "Listening on argilla:6900.",
                    "url": "http://localhost:6900",
                },
                {
                    "name": "parse-worker",
                    "purpose": "Turns a file into a document.",
                    "ok": None,
                    "detail": "Serves no port.",
                    "url": None,
                },
                {
                    "name": "redis",
                    "purpose": "The lock a stage takes.",
                    "ok": False,
                    "detail": "redis:6379 refused the connection.",
                    "url": None,
                },
            ],
            components={
                "database": {"ok": True, "detail": "Connected.", "metrics": {}},
                "ingestion": {
                    "ok": True,
                    "detail": "Documents held.",
                    "metrics": {"documents": 3, "upload_attempts": 5},
                },
            },
        )
    },
}


def opened(run_view, name: str) -> View:
    """Opens any page, whichever client it happens to hold.

    Every page also draws a configuration panel, which holds a client of
    its own, so every page is given one here: a panel that cannot reach the
    backend draws a caption instead of its controls, and a page test should
    be looking at the controls.
    """
    clients = OUTSIDE.get(name) or {"catalog_api": Answers(**answers())}
    return View(run_view(name, settings_api=Answers(**settings()), **clients), name)


# ── Every page ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", EVERY)
def test_a_page_renders(run_view, name) -> None:
    """Top to bottom, without raising."""
    assert opened(run_view, name).raised == []


@pytest.mark.parametrize("name", EVERY)
def test_a_page_draws_no_progress_bar(run_view, name) -> None:
    """A queue's depth is a number and a spinner, never a filling bar."""
    assert opened(run_view, name).progress_bars() == []


@pytest.mark.parametrize("name", LISTING)
def test_a_page_opens_an_analysis_fold_rather_than_spilling_it(run_view, name) -> None:
    """Everything beyond the headline figures is folded away on arrival.

    Two folds now: the analysis, and the configuration of the one service
    this page runs. Neither is what anybody arrives to do, and a page that
    opened with forty numbers on it is a page nobody reads.
    """
    assert opened(run_view, name).folds() == ["Analysis", "Configuration"]


@pytest.mark.parametrize("name", LISTING)
def test_every_page_is_laid_out_the_same_way(run_view, name) -> None:
    """Overview, the service, search, filters, the table. In that order.

    Topics carries a sixth: its deletion is over the whole model rather
    than over a row, so the box is on the page and not in a selection.
    """
    drawn = opened(run_view, name).panels()

    assert [drawn[0], *drawn[2:4]] == ["Overview", "Search", "Filters"], drawn
    assert drawn[4].lower().startswith(name), drawn
    assert drawn[5:] in ([], ["Delete"]), drawn


@pytest.mark.parametrize("name", LISTING)
def test_a_page_names_its_service_between_the_figures_and_the_search(
    run_view, name
) -> None:
    """So a reader finds the controls in the same place on every page."""
    drawn = opened(run_view, name).panels()

    assert drawn[1].lower().startswith(OWNED[name][:5]), drawn


@pytest.mark.parametrize("name", LISTING)
def test_a_page_shows_a_handful_of_figures_and_no_more(run_view, name) -> None:
    """The first section carries what is worth seeing, not everything."""
    assert len(opened(run_view, name).stats()) <= 5


@pytest.mark.parametrize("name", LISTING)
def test_every_figure_says_what_it_counts(run_view, name) -> None:
    """A figure cannot reach the page without a line saying what it is."""
    unexplained = [
        label
        for label, text in opened(run_view, name).explanations().items()
        if not text
    ]

    assert not unexplained


@pytest.mark.parametrize("name", EVERY)
def test_a_page_reports_an_unreachable_backend_rather_than_raising(
    run_view, name
) -> None:
    """Streamlit would otherwise replace the page with a stack trace."""
    down = requests.exceptions.ConnectionError("backend is down")
    clients = (
        {"catalog_api": Answers(**dict.fromkeys(answers(), down))}
        if name in LISTING
        else {
            "upload_api": Answers(counts=down),
            "health_api": Answers(
                reachable=(False, "down"), components={}, services=[]
            ),
        }
    )
    page = View(run_view(name, **clients), name)

    assert page.raised == []
    assert "Failed to load" in page.text() or "unreachable" in page.text().lower()


# ── One service per page ──────────────────────────────────────────────────


@pytest.mark.parametrize("name", LISTING)
def test_a_page_runs_its_own_stage_and_no_other(run_view, name) -> None:
    """The rule the whole layout is built on.

    Read off the controls themselves: every queue button is keyed with the
    stage it moves, so a page carrying a control for somebody else's stage
    says so in its own keys.
    """
    page = opened(run_view, name)
    verbs = ("start", "stop", "retry", "rerun", "discover")

    moved = {
        key.split("-")[1]
        for key in page.control_keys()
        if key and key.split("-")[0] in verbs
    }

    assert moved == {OWNED[name]}, page.control_keys()


@pytest.mark.parametrize("name", LISTING)
def test_a_page_can_run_its_stage_over_everything_at_once(run_view, name) -> None:
    """Each service is startable for the whole corpus, not row by row."""
    page = opened(run_view, name)

    assert page.button("Start all") or page.button("Fit all"), sorted(page.buttons())


@pytest.mark.parametrize("name", ("documents", "passages", "facts", "questions"))
def test_a_stage_that_queues_rows_offers_the_four_verbs(run_view, name) -> None:
    """Start, Stop, Retry and Redo, whichever page the stage is on."""
    offered = set(opened(run_view, name).buttons())

    assert {"Start all", "Stop", "Retry", "Redo all"} <= offered


def test_the_topics_page_offers_a_fit_rather_than_a_start(run_view) -> None:
    """A fit is all-or-nothing, so it has no per-topic form and no redo."""
    page = opened(run_view, "topics")

    assert page.button("Fit all")
    assert page.button("Redo all") is None


# ── Search, filters and the table ─────────────────────────────────────────


@pytest.mark.parametrize("name", LISTING)
def test_searching_is_a_separate_control_from_filtering(run_view, name) -> None:
    """Typing words is one thing; narrowing to a value another."""
    keys = opened(run_view, name).widget_keys()

    assert f"{name}-search" in keys
    assert {key for key in keys if key.startswith(f"{name}-")} - {
        f"{name}-search",
        f"{name}-field",
        f"{name}-page",
    }, keys


@pytest.mark.parametrize("name", LISTING)
def test_nothing_about_an_item_is_shown_until_one_is_picked(run_view, name) -> None:
    """A page opens as a list, and stays one until a row is clicked."""
    page = opened(run_view, name)

    assert "Field" not in page.tables()


@pytest.mark.parametrize("name", LISTING)
def test_picking_a_row_opens_everything_held_about_it(run_view, name) -> None:
    """One table of every field, rather than a second row of figures."""
    page = opened(run_view, name).select(0)

    assert page.raised == []
    assert "Field" in page.tables()


@pytest.mark.parametrize("name", LISTING)
def test_picking_a_row_adds_no_figures(run_view, name) -> None:
    """Below the table everything is a table. No boxes."""
    before = opened(run_view, name)
    after = before.select(0)

    assert set(after.stats()) == set(before.stats())


@pytest.mark.parametrize("name", LISTING)
def test_a_page_with_nothing_on_it_says_so(run_view, name) -> None:
    """An empty corpus is a sentence, not a stack trace or a blank."""
    page = View(
        run_view(
            name,
            catalog_api=Answers(
                **answers(
                    documents={"total": 0, "documents": []},
                    passages={"total": 0, "passages": []},
                    facts={"total": 0, "facts": []},
                    questions={"total": 0, "questions": []},
                    topics=[],
                    stage_status=lambda *a, **k: status(),
                )
            ),
        ),
        name,
    )

    assert page.raised == []
    assert "matches" in page.text() or "Fit the model" in page.text()


# ── Deletion ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ("passages", "facts", "questions"))
def test_a_page_whose_rows_cannot_be_deleted_offers_no_deletion(run_view, name) -> None:
    """A passage belongs to its document; a rejected question is evidence."""
    page = opened(run_view, name).select(0)

    assert not [key for key in page.control_keys() if key and key.startswith("danger-")]


@pytest.mark.parametrize(
    ("name", "label"),
    (("documents", "Delete document"), ("topics", "Delete all topics")),
)
def test_a_deletion_asks_before_it_does_anything(run_view, name, label) -> None:
    """The first click arms it; a second, separately labelled one acts."""
    page = opened(run_view, name).select(0)
    assert page.button("Yes, delete") is None

    armed = page.press(label)

    assert armed.button("Yes, delete")
    assert armed.button("Cancel")


def test_cancelling_a_deletion_removes_nothing(run_view) -> None:
    """Nothing is asked of the backend until the second click."""
    client = Answers(**answers())
    page = View(run_view("documents", catalog_api=client), "documents").select(0)

    page.press("Delete document").press("Cancel")

    assert not [one for one in client.asked if one[0] == "delete"]


def test_confirming_a_deletion_deletes(run_view) -> None:
    """And says what went."""
    client = Answers(**answers())
    page = View(run_view("documents", catalog_api=client), "documents").select(0)

    page.press("Delete document").press("Yes, delete")

    assert [one for one in client.asked if one[0] == "delete"]


def test_deleting_the_derived_data_keeps_the_document(run_view) -> None:
    """The two deletions are different operations, not one with a flag."""
    client = Answers(**answers())
    page = View(run_view("documents", catalog_api=client), "documents").select(0)

    page.press("Delete passages and facts").press("Yes, delete")

    assert [
        one for one in client.asked if one[0] == "delete" and one[2].get("derived_only")
    ]


# ── The pages outside the catalogue ───────────────────────────────────────


def test_the_upload_page_offers_a_file_picker(run_view) -> None:
    """The one page whose service runs in the request rather than a worker."""
    app = run_view("upload", **OUTSIDE["upload"])

    assert app.get("file_uploader")


def test_the_upload_page_uploads_every_chosen_file_at_once(run_view) -> None:
    """Its all-at-once is the picker taking several files."""
    app = run_view("upload", **OUTSIDE["upload"])

    assert app.get("file_uploader")[0].proto.multiple_files


def test_the_health_page_lists_every_component_the_backend_reports(run_view) -> None:
    """A component appears without a change here."""
    page = opened(run_view, "health")

    assert "Ingestion" in page.tables()
    assert "Database" in page.tables()


def test_the_health_page_shows_a_component_s_figures_when_it_is_picked(
    run_view,
) -> None:
    """Its own numbers, and what each counts."""
    page = opened(run_view, "health")
    page.app.session_state["health-table"] = {"selection": {"rows": [2], "columns": []}}
    picked = View(page.app.run(), "health")

    assert "Upload Attempts" in picked.tables()
    assert "accepted and refused" in picked.tables()


def test_the_health_page_lists_every_service_with_a_link_to_it(run_view) -> None:
    """The whole of compose, not the browsable half.

    A page called System health that listed only what has a web page could
    not answer "is Redis up", which is the question somebody has when a
    stage stops claiming rows.
    """
    page = opened(run_view, "health")

    assert "argilla" in page.tables()
    assert "redis" in page.tables()
    assert "http://localhost:6900" in page.tables()


def test_the_health_page_says_a_worker_is_neither_up_nor_down(run_view) -> None:
    """It reads a queue and serves no port, so a red mark would be a lie."""
    page = opened(run_view, "health")

    assert "no port" in page.tables()


def test_the_health_page_names_what_is_not_answering(run_view) -> None:
    """A table of twenty-three rows buries one failure."""
    page = opened(run_view, "health")

    assert "redis" in page.text()


def test_the_health_page_survives_a_backend_that_is_down(run_view) -> None:
    """A failed backend card makes every other card unknowable."""
    page = View(
        run_view(
            "health",
            health_api=Answers(
                reachable=(False, "refused"), components={}, services=[]
            ),
        ),
        "health",
    )

    assert page.raised == []
    assert "unreachable" in page.text()


# ── What each listing page put in its table ───────────────────────────────


def test_the_documents_page_lists_what_the_backend_returns(run_view) -> None:
    """The row, as a reader recognises it."""
    tables = opened(run_view, "documents").tables()

    assert "report.pdf" in tables
    assert "Risks in focus" in tables


def test_the_documents_page_can_narrow_to_one_parse_state(run_view) -> None:
    """Which is the filter the parsing page has."""
    client = Answers(**answers())
    page = View(run_view("documents", catalog_api=client), "documents")

    page.choose("documents-parse_status", "failed")

    assert [
        one
        for one in client.asked
        if one[0] == "documents" and one[2].get("parse_status") == "failed"
    ]


def test_the_passages_page_numbers_the_sentences_a_fact_would_cite(run_view) -> None:
    """The number in the first column is what a citation refers to."""
    tables = opened(run_view, "passages").select(0).tables()

    assert "Claims" in tables


def test_the_passages_page_runs_chunking_over_the_document_it_belongs_to(
    run_view,
) -> None:
    """Chunking queues over a document and replaces all of its passages."""
    page = opened(run_view, "passages").select(0)

    assert f"start-chunking-document-{SHA}" in page.control_keys()


def test_the_facts_page_runs_extraction_over_the_passage_it_came_from(
    run_view,
) -> None:
    """Extraction reads a passage and writes all of its facts together."""
    page = opened(run_view, "facts").select(0)

    assert "start-extraction-passage-11" in page.control_keys()


def test_the_questions_page_lists_what_the_backend_returns(run_view) -> None:
    """The question and the answer it is scored against."""
    tables = opened(run_view, "questions").tables()

    assert "48 hours" in tables


def test_accepting_a_question_writes_the_status_and_nothing_else(run_view) -> None:
    """A person overruling a gate changes the verdict, not the question."""
    client = Answers(**answers(questions={"total": 1, "questions": [QUESTION_DRAFT]}))
    page = View(run_view("questions", catalog_api=client), "questions").select(0)

    page.press("Accept")

    assert [
        one
        for one in client.asked
        if one[0] == "decide_question" and one[1][1] == "accepted"
    ]


def test_the_questions_page_says_when_a_cited_fact_no_longer_holds(run_view) -> None:
    """A question resting on a rejected fact is not a question any more."""
    page = View(
        run_view(
            "questions",
            catalog_api=Answers(
                **answers(
                    question={
                        "question": QUESTION,
                        "sources": [changed(QUESTION_SOURCE, validated=False)],
                        "thread": [],
                    }
                )
            ),
        ),
        "questions",
    ).select(0)

    assert "no longer passes its checks" in page.text()
