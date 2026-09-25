"""The Assessment page, run against a scripted backend.

This page had no test at all, and shipped broken: its Configuration panel
called `GET /settings/assessment` for a service the route's own Literal did
not list, so the panel answered 422 and the page showed an error. The rules
it shares with every other page are in test_views.py; this covers what only
the evaluation phase has - two machine verdicts on one row, and the pair
where they differ.
"""

from __future__ import annotations

import pytest
from pages import ASSESSMENT, ASSESSMENT_PLAN, View, changed

pytestmark = pytest.mark.frontend


def page_of(*assessments: dict) -> dict:
    """One page of verdicts, and the total behind it."""
    return {"total": len(assessments), "assessments": list(assessments)}


def verdicts(drawn: View):
    """The table of verdicts, which is not the first table on the page.

    The Analysis fold draws the findings above it, so an index here would
    be reading the queue counts and calling them the verdicts.
    """
    for frame in drawn.app.dataframe:
        if "Judge" in getattr(frame.value, "columns", ()):
            return frame.value
    raise AssertionError("no table on the page carries the judge's verdict")


@pytest.fixture
def page(open_view):
    """Runs the Assessment view against whatever the backend says."""

    def run(**replaced) -> View:
        """Runs the page with these answers replacing the defaults."""
        return open_view("assessment", **replaced)

    return run


def test_the_page_renders(page) -> None:
    """The one thing nothing checked, and the reason it shipped broken."""
    assert not page().app.exception


def test_every_artefact_kind_is_a_tab(page) -> None:
    """Facts, topics and questions, each readable on its own.

    A tab rather than a dropdown in the filter panel, because "show me the
    questions only" is what somebody arrives at this page wanting, and a
    control buried three panels down does not read as an answer to it.
    """
    offered = page().app.segmented_control("assessment-tab").options

    assert offered == ["All artefacts", "Facts", "Topics", "Questions"]


def test_the_page_opens_on_every_artefact(page) -> None:
    """The tab strip opens on all of them.

    Arriving at a page silently narrowed to a third of the corpus is
    arriving at a lie about how much has been judged.
    """
    assert page().app.segmented_control("assessment-tab").value == "All artefacts"


def test_every_kind_can_be_ticked_for_judging(page) -> None:
    """Which is a different question from which kind is being looked at.

    Reading one corpus while queueing another is a reasonable thing to
    want, and one control for both would make it impossible.
    """
    ticked = {
        one.key: one.value
        for one in page().app.checkbox
        if one.key.startswith("assessment-judge-")
    }

    assert set(ticked) == {
        "assessment-judge-fact",
        "assessment-judge-topic",
        "assessment-judge-question",
    }
    assert all(ticked.values()), "the page opens agreeing with ASSESSMENT_KINDS"


def test_a_kind_the_deployment_does_not_judge_opens_unticked(page) -> None:
    """So the page says what the worker would actually do."""
    drawn = page(
        assessment_plan={**ASSESSMENT_PLAN, "kinds": ["question"]},
    )
    ticked = {
        one.key: one.value
        for one in drawn.app.checkbox
        if one.key.startswith("assessment-judge-")
    }

    assert ticked["assessment-judge-question"] is True
    assert ticked["assessment-judge-fact"] is False


def test_every_metric_is_offered_as_a_filter(page) -> None:
    """So a reader can ask what one named judgement caught."""
    offered = page().app.selectbox("assessment-metric").options

    assert offered == [
        "All refused by",
        "hallucination",
        "qa_correctness",
        "relevance",
        "summarization",
        "refusal",
        "conciseness",
        "toxicity",
    ]


def test_the_disagreements_are_counted_where_a_reader_looks_first(page) -> None:
    """The only figure on this page anybody should act on."""
    drawn = page()

    assert any(
        one.value == "1" and "Disagreements" in one.label for one in drawn.app.metric
    )


def test_a_refused_row_names_the_judgement_that_refused_it(page) -> None:
    """`hallucinated` is not a word to leave a reader to infer."""
    shown = verdicts(page())

    assert "Hallucination" in str(shown["Refused by"].tolist())


def test_a_disagreement_is_marked_as_one(page) -> None:
    """Kept by the pipeline, refused by the judge."""
    shown = verdicts(page())

    assert shown["Disagrees"].tolist() == ["yes"]
    assert shown["Pipeline"].tolist() == ["accepted"]
    assert shown["Judge"].tolist() == ["refused"]


def test_an_artefact_nobody_has_judged_reads_as_not_yet(page) -> None:
    """Rather than as one the judge declined to approve.

    A blank or a `no` there would say the judge found something wrong with
    a row it has never been asked about.
    """
    waiting = changed(ASSESSMENT, approved=None, metrics=[], assessed_at=None)
    shown = verdicts(page(assessments=page_of(waiting)))

    assert shown["Judge"].tolist() == ["not yet"]


def test_the_page_says_so_when_the_phase_is_off(page) -> None:
    """A deployment that has not turned this on has not got a bug."""
    drawn = page(assessment_plan={**ASSESSMENT_PLAN, "enabled": False})

    assert any("ASSESSMENT_ENABLED" in one.value for one in drawn.app.info)


def test_the_judge_is_named_beside_the_figures(page) -> None:
    """It should not be the model that wrote what it is judging."""
    drawn = page()

    assert any(one.value == "ollama_chat/qwen3:14b" for one in drawn.app.metric)


def test_nothing_matching_is_a_caption_rather_than_an_empty_table(page) -> None:
    """The shape every other listing page answers an empty filter with."""
    drawn = page(assessments=page_of())

    assert not drawn.app.exception
    assert any("Nothing matches" in one.value for one in drawn.app.caption)


def test_a_backend_that_is_down_is_a_message_and_not_a_traceback(page) -> None:
    """`page.render` catches it; this is the page under that guard."""
    drawn = page(assessment_plan=RuntimeError("the backend is unreachable"))

    assert not drawn.app.exception
    assert any("Failed to load" in one.value for one in drawn.app.error)
