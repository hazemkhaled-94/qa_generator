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
from pages import ASSESSMENT, ASSESSMENT_PLAN, changed

pytestmark = pytest.mark.frontend


def page_of(*assessments: dict) -> dict:
    """One page of verdicts, and the total behind it."""
    return {"total": len(assessments), "assessments": list(assessments)}


#: The view this module is about. `page` in conftest.py reads it.
VIEW = "assessment"


def test_the_page_renders(page) -> None:
    """The one thing nothing checked, and the reason it shipped broken."""
    assert page().raised == []


def test_every_artefact_kind_is_a_tab(page) -> None:
    """Facts, topics and questions, each readable on its own.

    A tab rather than a dropdown in the filter panel, because "show me the
    questions only" is what somebody arrives at this page wanting, and a
    control buried three panels down does not read as an answer to it.
    """
    offered = page().options("assessment-tab")

    assert offered == ["All artefacts", "Facts", "Topics", "Questions"]


def test_the_page_opens_on_every_artefact(page) -> None:
    """The tab strip opens on all of them.

    Arriving at a page silently narrowed to a third of the corpus is
    arriving at a lie about how much has been judged.
    """
    assert page().chosen("assessment-tab") == "All artefacts"


def test_every_kind_can_be_ticked_for_judging(page) -> None:
    """Which is a different question from which kind is being looked at.

    Reading one corpus while queueing another is a reasonable thing to
    want, and one control for both would make it impossible.
    """
    ticked = page().judging()

    assert set(ticked) == {"fact", "topic", "question"}
    assert all(ticked.values()), "the page opens agreeing with ASSESSMENT_KINDS"


def test_a_kind_the_deployment_does_not_judge_opens_unticked(page) -> None:
    """So the page says what the worker would actually do."""
    drawn = page(
        assessment_plan={**ASSESSMENT_PLAN, "kinds": ["question"]},
    )
    ticked = drawn.judging()

    assert ticked["question"] is True
    assert ticked["fact"] is False


def test_every_metric_is_offered_as_a_filter(page) -> None:
    """So a reader can ask what one named judgement caught."""
    offered = page().options("assessment-metric")

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
        label
        for label, value in drawn.stats().items()
        if "Disagreements" in label and value == "1"
    )


def test_a_refused_row_names_the_judgement_that_refused_it(page) -> None:
    """`hallucinated` is not a word to leave a reader to infer."""
    shown = page().verdicts()

    assert "Hallucination" in str(shown["Refused by"].tolist())


def test_a_disagreement_is_marked_as_one(page) -> None:
    """Kept by the pipeline, refused by the judge."""
    shown = page().verdicts()

    assert shown["Disagrees"].tolist() == ["yes"]
    assert shown["Pipeline"].tolist() == ["accepted"]
    assert shown["Judge"].tolist() == ["refused"]


def test_an_artefact_nobody_has_judged_reads_as_not_yet(page) -> None:
    """Rather than as one the judge declined to approve.

    A blank or a `no` there would say the judge found something wrong with
    a row it has never been asked about.
    """
    waiting = changed(ASSESSMENT, approved=None, metrics=[], assessed_at=None)
    shown = page(assessments=page_of(waiting)).verdicts()

    assert shown["Judge"].tolist() == ["not yet"]


def test_the_page_says_so_when_the_phase_is_off(page) -> None:
    """A deployment that has not turned this on has not got a bug."""
    drawn = page(assessment_plan={**ASSESSMENT_PLAN, "enabled": False})

    assert any("ASSESSMENT_ENABLED" in one for one in drawn.notes())


def test_the_judge_is_named_beside_the_figures(page) -> None:
    """It should not be the model that wrote what it is judging."""
    drawn = page()

    assert "ollama_chat/qwen3:14b" in drawn.stats().values()


def test_nothing_matching_is_a_caption_rather_than_an_empty_table(page) -> None:
    """The shape every other listing page answers an empty filter with."""
    drawn = page(assessments=page_of())

    assert drawn.raised == []
    assert any("Nothing matches" in one for one in drawn.captions())


def test_a_backend_that_is_down_is_a_message_and_not_a_traceback(page) -> None:
    """`page.render` catches it; this is the page under that guard."""
    drawn = page(assessment_plan=RuntimeError("the backend is unreachable"))

    assert drawn.raised == []
    assert any("Failed to load" in one for one in drawn.errors())
