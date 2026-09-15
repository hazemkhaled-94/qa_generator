"""The Topics page, run against a scripted backend.

A page that raises renders a Streamlit traceback and nothing else, so every
test here asserts the page rendered as well as what it said.
"""

from __future__ import annotations

import pytest
import requests
from conftest import Answers
from pages import TopicsPage, fit, language_fit, topic

pytestmark = pytest.mark.frontend

#: What the backend answers when it is down.
UNREACHABLE = requests.exceptions.ConnectionError("connection refused")


@pytest.fixture
def page(run_view):
    """Runs the Topics view against whatever the backend is scripted to say."""

    def run(**changed) -> TopicsPage:
        """Runs the page with these answers replacing the defaults."""
        return TopicsPage(
            run_view("topics", catalog_api=Answers(**TopicsPage.answers(**changed)))
        )

    return run


# ── An empty deployment ───────────────────────────────────────────────────


def test_a_deployment_with_no_topics_says_what_to_do_first(page) -> None:
    """The first thing a new deployment shows."""
    drawn = page(topics=[], topic_fit=fit(languages=[], status=None, topics=0))

    assert not drawn.raised, drawn.raised
    assert "No topics yet" in drawn.text(), drawn.text()
    assert "Chunk some documents" in drawn.text()


def test_the_fit_control_is_offered_with_no_topics_stored(page) -> None:
    """There is nothing to refit, and that is when a first fit is asked for."""
    drawn = page(topics=[], topic_fit=fit(languages=[], status=None, topics=0))

    assert drawn.button("Fit the corpus") is not None
    assert not drawn.button("Fit the corpus").disabled


def test_a_backend_that_is_down_is_reported_rather_than_raised(page) -> None:
    """Every call fails; the reader is told, not shown a stack."""
    drawn = page(
        topics=UNREACHABLE,
        topic_fit=UNREACHABLE,
        stage_status=UNREACHABLE,
        topic_visualisation=UNREACHABLE,
    )

    assert not drawn.raised, drawn.raised


# ── The stored model ──────────────────────────────────────────────────────


def test_the_model_figures_are_drawn_from_the_fit(page) -> None:
    """One figure each, and none of them computed on the page."""
    drawn = page()

    figures = drawn.metrics()
    assert figures["Topics"] == "1", figures
    assert figures["Language models"] == "1", figures
    assert figures["Memberships"] == "60", figures


def test_the_model_age_reads_as_never_before_any_fit(page) -> None:
    """A date of None must not be parsed as one."""
    drawn = page(
        topics=[], topic_fit=fit(languages=[language_fit(fitted_at=None)], topics=0)
    )

    assert not drawn.raised, drawn.raised
    assert drawn.metrics()["Model age"] == "never", drawn.metrics()


def test_every_topic_is_tabled_with_its_terms(page) -> None:
    """One row per topic per language."""
    drawn = page()

    tabled = drawn.tables()
    assert "lieferung" in tabled, tabled
    assert "versand" in tabled


def test_a_topic_with_no_label_reads_as_unnamed_rather_than_blank(page) -> None:
    """Both columns are nullable, and an empty cell reads as a defect."""
    drawn = page(topics=[topic(label=None, labelled_by=None)])

    assert not drawn.raised, drawn.raised
    assert "—" in drawn.tables(), drawn.tables()


def test_a_topic_with_no_language_still_renders(page) -> None:
    """The column is nullable, and a page that assumes otherwise breaks."""
    drawn = page(topics=[topic(language=None)])

    assert not drawn.raised, drawn.raised


def test_a_topic_holding_no_passage_does_not_divide_by_zero(page) -> None:
    """Facts-per-passage over zero passages used to raise."""
    drawn = page(
        topics=[topic(passages=0, dominant_passages=0, validated_facts=0, documents=0)]
    )

    assert not drawn.raised, drawn.raised
    assert "0.0" in drawn.tables(), drawn.tables()


# ── Model health ──────────────────────────────────────────────────────────


def test_a_healthy_model_reports_every_check_as_ok(page) -> None:
    """A row reading OK needs nothing done."""
    drawn = page()

    assert "Attention" not in drawn.tables(), drawn.tables()


def test_a_model_whose_passages_were_rechunked_is_flagged(page) -> None:
    """The memberships went with them, so nothing is topic-weighted."""
    drawn = page(
        topic_fit=fit(languages=[language_fit(memberships=0, live_passages=0)])
    )

    assert "Attention" in drawn.tables()
    assert "re-chunked" in drawn.tables(), drawn.tables()


def test_a_corpus_that_has_grown_since_the_fit_is_flagged(page) -> None:
    """The topics describe the older corpus."""
    drawn = page(topic_fit=fit(languages=[language_fit(live_passages=140)]))

    assert "Attention" in drawn.tables()
    assert "has changed since this model was fitted" in drawn.tables()


def test_unplaced_passages_name_both_settings_that_cause_them(page) -> None:
    """A passage can be unplaced by the vocabulary filter or by the floor."""
    drawn = page(topic_fit=fit(languages=[language_fit(passages_without_topics=12)]))

    tabled = drawn.tables()
    assert "TOPIC_NO_BELOW" in tabled, tabled
    assert "TOPIC_MIN_WEIGHT" in tabled, tabled


def test_passages_with_no_language_are_flagged(page) -> None:
    """No model covers them, whatever the fit did."""
    drawn = page(topic_fit=fit(passages_without_language=7))

    assert "7 without one" in drawn.tables(), drawn.tables()
    assert "Attention" in drawn.tables()


def test_a_failed_fit_is_reported_beside_the_working_topics(page) -> None:
    """The topics above are from an earlier fit, and the page says so."""
    drawn = page(topic_fit=fit(error="the frequency filter left no terms"))

    assert "frequency filter" in drawn.tables(), drawn.tables()
    assert "Retry below" in drawn.tables()


def test_a_topic_excluded_by_hand_is_reported_as_a_decision(page) -> None:
    """Not a fault: what counts as a subject belongs to the corpus."""
    drawn = page(topics=[topic(include_in_coverage=False)])

    assert "1 excluded" in drawn.tables(), drawn.tables()
    assert "a deliberate choice" in drawn.tables()


def test_every_setting_the_page_names_is_one_that_exists(page) -> None:
    """The Topics figure used to name TOPIC_COUNT, which nothing reads."""
    from topic_modelling.config import Settings

    drawn = page()
    said = " ".join(drawn.explanations().values()) + drawn.tables()

    assert "TOPIC_NUM_TOPICS" in said, said
    named = {word.strip(".,") for word in said.split() if word.startswith("TOPIC_")}
    known = {f"TOPIC_{one.upper()}" for one in Settings.__dataclass_fields__} | {
        "TOPIC_NUM_TOPICS",
        "TOPIC_LANGUAGE_NAMES",
    }
    assert named <= known, f"the page names settings nothing reads: {named - known}"


# ── The topic map ─────────────────────────────────────────────────────────


def test_a_language_with_no_figure_says_how_to_get_one(page) -> None:
    """It is drawn by a fit, so the answer is to fit again."""
    drawn = page(topic_visualisation=None)

    assert "No map for de yet" in drawn.text(), drawn.text()


def test_a_drawn_figure_is_embedded(page) -> None:
    """The page hands the stored page to the browser as it is."""
    drawn = page(topic_visualisation="<html>the de figure</html>")

    assert not drawn.raised, drawn.raised
    assert drawn.embedded(), "the figure was not embedded"


def test_the_map_explains_what_the_reader_is_looking_at(page) -> None:
    """A pyLDAvis figure is unreadable without its legend."""
    drawn = page(topic_visualisation="<html>a figure</html>")

    assert "Overlapping circles" in drawn.text(), drawn.text()


# ── One topic ─────────────────────────────────────────────────────────────


def test_the_chosen_topic_reports_its_own_figures(page) -> None:
    """Measurements of one topic within its own language model."""
    drawn = page()

    figures = drawn.metrics()
    assert figures["Passages"] == "40", figures
    assert figures["Documents"] == "3", figures
    assert figures["Mean weight"] == "0.42", figures


def test_the_chosen_topic_shows_what_a_label_is_matched_on(page) -> None:
    """A refit carries a label by top terms and nothing else."""
    drawn = page()

    assert "Top terms: lieferung, versand, transport" in drawn.text(), drawn.text()


def test_naming_a_topic_sends_the_label_and_the_coverage_choice(page, run_view) -> None:
    """The one control on this page that changes a row."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    drawn.type_label(1, "Shipping").press("Save")

    sent = [one for one in client.asked if one[0] == "describe_topic"]
    assert sent == [("describe_topic", (1, "Shipping", True), {})], sent


def test_a_label_of_only_spaces_is_sent_as_no_label(page, run_view) -> None:
    """Rather than a topic named with a blank."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    drawn.type_label(1, "   ").press("Save")

    sent = [one for one in client.asked if one[0] == "describe_topic"]
    assert sent == [("describe_topic", (1, None, True), {})], sent


def test_taking_a_topic_out_of_coverage_sends_the_choice(page, run_view) -> None:
    """It keeps its passages; only the reporting changes."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    drawn.clear_coverage(1).press("Save")

    sent = [one for one in client.asked if one[0] == "describe_topic"]
    assert sent and sent[0][1][2] is False, sent


# ── Fitting ───────────────────────────────────────────────────────────────


def test_fitting_the_corpus_queues_a_fit(page, run_view) -> None:
    """Acts on the whole corpus: there is no per-topic fit."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    drawn.press("Fit the corpus")

    assert ("stage_action", ("topics", "discover", None), {}) in client.asked


def test_the_fit_control_is_dead_while_a_fit_is_queued(page) -> None:
    """A control that would do nothing is greyed rather than hidden."""
    drawn = page(
        stage_status={"stage": "topics", "working": False, "rows": {"pending": 1}}
    )

    assert drawn.button("Fit the corpus").disabled
    assert drawn.button("Stop") is not None and not drawn.button("Stop").disabled


def test_the_fit_control_is_dead_while_a_fit_runs(page) -> None:
    """Only one worker can claim a fit, and one already has."""
    drawn = page(
        stage_status={"stage": "topics", "working": True, "rows": {"in_progress": 1}}
    )

    assert drawn.button("Fit the corpus").disabled


def test_retry_is_dead_when_no_fit_has_failed(page) -> None:
    """There is nothing to return to the queue."""
    drawn = page()

    assert drawn.button("Retry").disabled


def test_retry_is_live_when_a_fit_has_failed(page) -> None:
    """The one control that recovers a failure."""
    drawn = page(
        stage_status={"stage": "topics", "working": False, "rows": {"failed": 1}}
    )

    assert not drawn.button("Retry").disabled


def test_the_page_says_a_fit_cannot_be_narrowed_to_one_topic(page) -> None:
    """Every topic is fitted jointly, so the reader is told why."""
    drawn = page()

    assert "cannot be started, stopped or refitted on its own" in drawn.text()


def test_the_page_says_what_the_stored_model_was_fitted_over(page) -> None:
    """Compared against the live count in Model health above."""
    drawn = page()

    said = drawn.text()
    assert "100 passages" in said, said
    assert "800-term vocabulary" in said, said


# ── Deleting ──────────────────────────────────────────────────────────────


def test_deleting_asks_before_it_does_anything(page, run_view) -> None:
    """Nothing is removed until a second, separate press."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    armed = drawn.press("Delete all topics")

    assert not any(one[0] == "delete_topics" for one in client.asked)
    assert "Delete every topic" in armed.text(), armed.text()


def test_confirming_the_deletion_removes_the_topics(page, run_view) -> None:
    """And says what went, because nothing else stores a label."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    done = drawn.press("Delete all topics").press("Yes, delete the topics")

    assert any(one[0] == "delete_topics" for one in client.asked)
    # A toast, because the rerun that refreshes the listing discards
    # everything this run wrote to the page itself.
    said = " ".join(done.toasts())
    assert "Removed 2 topic(s)" in said, said
    assert "1 label(s)" in said


def test_cancelling_the_deletion_removes_nothing(page, run_view) -> None:
    """The armed state is cleared and the page goes back to itself."""
    client = Answers(**TopicsPage.answers())
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    back = drawn.press("Delete all topics").press("Cancel")

    assert not any(one[0] == "delete_topics" for one in client.asked)
    assert "Delete every topic" not in back.text()


def test_the_deletion_warning_is_different_when_there_is_nothing_to_delete(
    page, run_view
) -> None:
    """It still clears a queued or failed fit, and says so."""
    client = Answers(
        **TopicsPage.answers(
            topics=[], topic_fit=fit(languages=[], status=None, topics=0)
        )
    )
    drawn = TopicsPage(run_view("topics", catalog_api=client))

    armed = drawn.press("Delete all topics")

    assert "There are no topics" in armed.text(), armed.text()
