"""The Topics page, run against a scripted backend.

The rules it shares with every other page are in test_views.py. This covers
what only topic modelling has: a model fitted jointly, so there is nothing
to run or delete one topic at a time, and a name a person gives one.
"""

from __future__ import annotations

import pytest
from conftest import Answers
from pages import FIT, LANGUAGE_FIT, TOPIC, View, answers, changed, status

pytestmark = pytest.mark.frontend


def fit(languages: list[dict] | None = None, **replaced) -> dict:
    """The model's state, with whatever a test needs changed."""
    return {
        **FIT,
        "languages": [LANGUAGE_FIT] if languages is None else languages,
        **replaced,
    }


@pytest.fixture
def page(open_view):
    """Runs the Topics view against whatever the backend is scripted to say."""

    def run(**replaced) -> View:
        """Runs the page with these answers replacing the defaults."""
        return open_view("topics", **replaced)

    return run


# ── An empty deployment ───────────────────────────────────────────────────


def test_a_deployment_with_no_topics_says_what_to_do_first(page) -> None:
    """Rather than an empty table and no explanation."""
    drawn = page(topics=[], topic_fit=fit(languages=[], topics=0))

    assert drawn.raised == []
    assert "Fit the model" in drawn.text()


def test_the_fit_control_is_offered_with_no_topics_stored(page) -> None:
    """The one thing to do on an empty deployment has to be reachable."""
    drawn = page(
        topics=[],
        topic_fit=fit(languages=[], topics=0),
        stage_status=lambda *a, **k: status(),
    )

    assert not drawn.button("Fit all").disabled


# ── The stored model ──────────────────────────────────────────────────────


def test_the_model_figures_are_drawn_from_the_fit(page) -> None:
    """One topic, one language model, and what it placed."""
    figures = page().stats()

    assert figures["Topics"] == "1"
    assert figures["Language models"] == "1"
    assert figures["Passages placed"] == "25 (25%)"


def test_the_model_age_reads_as_never_before_any_fit(page) -> None:
    """Rather than as a date nothing produced."""
    figures = page(topic_fit=fit(languages=[language_never()])).stats()

    assert figures["Fitted"] == "never"


def language_never() -> dict:
    """One language whose model has never been fitted."""
    return changed(LANGUAGE_FIT, fitted_at=None)


def test_every_topic_is_tabled_with_its_terms(page) -> None:
    """The terms are what a topic is, before anybody names it."""
    tables = page().tables()

    assert "lieferung" in tables
    assert "versand" in tables


def test_a_topic_with_no_label_reads_as_unnamed_rather_than_blank(page) -> None:
    """An empty cell is indistinguishable from a bug."""
    assert "—" in page().tables()


def test_a_topic_with_no_language_still_renders(page) -> None:
    """A row the model produced without one should not break the page."""
    drawn = page(topics=[changed(TOPIC, language=None)])

    assert drawn.raised == []


def test_a_topic_holding_no_passage_does_not_divide_by_zero(page) -> None:
    """Facts-per-passage over nothing is 0, not a crash."""
    drawn = page(
        topics=[changed(TOPIC, passages=0, dominant_passages=0, validated_facts=0)]
    )

    assert drawn.raised == []
    assert "0.0" in drawn.tables()


# ── Model health ──────────────────────────────────────────────────────────


def test_a_healthy_model_reports_every_check_as_ok(page) -> None:
    """Nothing to do is a thing the page has to be able to say."""
    tables = page().tables()

    assert "Attention" not in tables
    assert "OK" in tables


def test_a_model_whose_passages_were_rechunked_is_flagged(page) -> None:
    """Every membership went with the passages; the topics hold nothing."""
    tables = page(
        topic_fit=fit(languages=[changed(LANGUAGE_FIT, memberships=0)])
    ).tables()

    assert "hold no passages" in tables


def test_a_corpus_that_has_grown_since_the_fit_is_flagged(page) -> None:
    """The topics still describe the older corpus."""
    tables = page(
        topic_fit=fit(languages=[changed(LANGUAGE_FIT, live_passages=180)])
    ).tables()

    assert "corpus has changed" in tables


def test_unplaced_passages_name_both_settings_that_cause_them(page) -> None:
    """A reader has to know which knob to turn."""
    tables = page(
        topic_fit=fit(languages=[changed(LANGUAGE_FIT, passages_without_topics=7)])
    ).tables()

    assert "TOPIC_NO_BELOW" in tables
    assert "TOPIC_MIN_WEIGHT" in tables


def test_passages_with_no_language_are_flagged(page) -> None:
    """They belong to no model, so nothing topic-weighted covers them."""
    tables = page(topic_fit=fit(passages_without_language=4)).tables()

    assert "too short for the detector" in tables


def test_a_failed_fit_is_reported_beside_the_working_topics(page) -> None:
    """The topics shown are from an earlier fit, which has to be said."""
    tables = page(topic_fit=fit(error="gensim exploded")).tables()

    assert "gensim exploded" in tables


def test_a_topic_excluded_by_hand_is_reported_as_a_decision(page) -> None:
    """Not as a fault: what counts as a subject belongs to the corpus."""
    tables = page(topics=[changed(TOPIC, include_in_coverage=False)]).tables()

    assert "excluded by hand" in tables


def test_every_setting_the_page_names_is_one_that_exists(page) -> None:
    """The Topics figure used to name TOPIC_COUNT, which nothing reads."""
    from topic_modelling.config import Settings

    drawn = page(
        topic_fit=fit(languages=[changed(LANGUAGE_FIT, passages_without_topics=7)])
    )
    said = " ".join(drawn.explanations().values()) + drawn.tables()

    named = {word.strip(".,") for word in said.split() if word.startswith("TOPIC_")}
    known = {f"TOPIC_{one.upper()}" for one in Settings.__dataclass_fields__} | {
        "TOPIC_NUM_TOPICS",
        "TOPIC_LANGUAGE_NAMES",
    }
    assert named, said
    assert named <= known, f"the page names settings nothing reads: {named - known}"


# ── The topic map ─────────────────────────────────────────────────────────


def test_a_language_with_no_figure_says_how_to_get_one(page) -> None:
    """It is drawn by a fit, so a model fitted earlier has none."""
    assert "A fit draws one" in page(topic_visualisation=None).text()


def test_a_drawn_figure_is_embedded(page) -> None:
    """The figure is a self-contained page, so it goes in an iframe."""
    drawn = page(topic_visualisation="<html>a figure</html>")

    assert drawn.embedded()


# ── One topic ─────────────────────────────────────────────────────────────


def test_the_chosen_topic_reports_everything_held_about_it(page) -> None:
    """Every field, as one table rather than as a second row of figures."""
    tables = page().select(0).tables()

    assert "Mean weight" in tables
    assert "Dominant in" in tables
    assert "Validated facts" in tables


def test_naming_a_topic_sends_the_label_and_the_coverage_choice(open_view) -> None:
    """Both, because one PATCH carries both."""
    client = Answers(**answers())
    page = open_view("topics", client).select(0)

    page.type_in(f"topics-label-{TOPIC['id']}", "Shipping").press("Save")

    assert ("describe_topic", (TOPIC["id"], "Shipping", True), {}) in client.asked


def test_a_label_of_only_spaces_is_sent_as_no_label(open_view) -> None:
    """Rather than stored as a name made of whitespace."""
    client = Answers(**answers())
    page = open_view("topics", client).select(0)

    page.type_in(f"topics-label-{TOPIC['id']}", "   ").press("Save")

    assert ("describe_topic", (TOPIC["id"], None, True), {}) in client.asked


def test_taking_a_topic_out_of_coverage_sends_the_choice(open_view) -> None:
    """A topic that is not a subject stops being counted as a gap."""
    client = Answers(**answers())
    page = open_view("topics", client).select(0)

    page.uncheck(f"topics-coverage-{TOPIC['id']}").press("Save")

    assert ("describe_topic", (TOPIC["id"], None, False), {}) in client.asked


# ── Fitting ───────────────────────────────────────────────────────────────


def test_fitting_the_corpus_queues_a_fit(open_view) -> None:
    """The page asks; the worker does it."""
    client = Answers(**answers(stage_status=lambda *a, **k: status(new=2)))

    open_view("topics", client).press("Fit all")

    assert ("stage_action", ("topics", "discover", None), {}) in client.asked


def test_the_fit_control_is_dead_while_a_fit_is_queued(page) -> None:
    """Asking twice would queue one fit behind another."""
    drawn = page(stage_status=lambda *a, **k: status(pending=1))

    assert drawn.button("Fit all").disabled


def test_the_fit_control_is_dead_while_a_fit_runs(page) -> None:
    """Same reason, one state later."""
    drawn = page(stage_status=lambda *a, **k: status(in_progress=1))

    assert drawn.button("Fit all").disabled


def test_retry_is_dead_when_no_fit_has_failed(page) -> None:
    """A control that would do nothing is greyed rather than hidden."""
    drawn = page(stage_status=lambda *a, **k: status(modelled=2))

    assert drawn.button("Retry").disabled


def test_retry_is_live_when_a_fit_has_failed(page) -> None:
    """And says so."""
    drawn = page(stage_status=lambda *a, **k: status(failed=1))

    assert not drawn.button("Retry").disabled


def test_the_page_says_a_fit_cannot_be_narrowed_to_one_topic(page) -> None:
    """A topic is one column of a model fitted jointly."""
    assert "no per-topic delete" in page().html()


# ── Deleting ──────────────────────────────────────────────────────────────


def test_confirming_the_deletion_removes_the_topics(open_view) -> None:
    """And says what went with them."""
    client = Answers(**answers())

    page = open_view("topics", client).press("Delete all topics")
    page.press("Yes, delete")

    assert [one for one in client.asked if one[0] == "delete_topics"]


def test_cancelling_the_deletion_removes_nothing(open_view) -> None:
    """Nothing is asked of the backend until the second click."""
    client = Answers(**answers())

    page = open_view("topics", client).press("Delete all topics")
    page.press("Cancel")

    assert not [one for one in client.asked if one[0] == "delete_topics"]


def test_the_deletion_warning_is_different_when_there_is_nothing_to_delete(
    page,
) -> None:
    """Deleting no topics still clears a queued or failed fit."""
    drawn = page(topics=[], topic_fit=fit(languages=[], topics=0)).press(
        "Delete all topics"
    )

    assert "There are no topics" in drawn.text()
