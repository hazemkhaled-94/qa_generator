"""The /topics routes, against the database and the object store.

Every route reads or writes the queue and none of them works it, so what
these cover is the status code, the body and the row each one leaves.
"""

from __future__ import annotations

import pytest
from topics import TopicsApi, TopicStore, fitting

from database.qa_generator import Status

pytestmark = pytest.mark.integration


@pytest.fixture
def api(client, engine) -> tuple[TopicsApi, TopicStore]:
    """The routes and the store behind them, both empty."""
    return TopicsApi(client), TopicStore(engine)


@pytest.fixture
def modelled(api):
    """One German model of two topics, one passage in the first."""
    routes, store = api
    store.given_passages(de=2)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))
    return routes, store


# ── The queue ─────────────────────────────────────────────────────────────


def test_the_queue_is_empty_before_anything_is_asked_for(api) -> None:
    """The first thing a new deployment reports."""
    routes, _ = api

    answered = routes.status()

    assert answered["stage"] == "topics"
    assert answered["working"] is False
    assert answered["rows"] == {}


def test_discovering_queues_a_fit_and_answers_at_once(api) -> None:
    """202: the worker picks it up on its next poll."""
    routes, _ = api

    answered = routes.discover()

    assert answered.status_code == 202
    body = answered.json()
    assert body["fit"] > 0
    assert "replace" in body["detail"]
    assert routes.status()["rows"] == {Status.PENDING: 1}


def test_discovering_twice_queues_one_fit(api) -> None:
    """The second ask replaces the first."""
    routes, _ = api

    routes.discover()
    routes.discover()

    assert routes.status()["rows"] == {Status.PENDING: 1}


def test_stopping_takes_a_queued_fit_off_the_queue(api) -> None:
    """The request is deleted, so no state is left behind."""
    routes, _ = api
    routes.discover()

    answered = routes.stop()

    # 200, not the 202 the shared stage router answers: these routes are this
    # stage's own, because a fit is asked for rather than started.
    assert answered.status_code == 200
    assert answered.json()["rows"] == 1
    assert routes.status()["rows"] == {}


def test_stopping_with_nothing_queued_answers_zero(api) -> None:
    """A verb that did nothing says so rather than refusing."""
    routes, _ = api

    assert routes.stop().json()["rows"] == 0


def test_retrying_returns_a_failed_fit_to_the_queue(api) -> None:
    """A failure is always recoverable."""
    routes, store = api
    failed = store.request()
    store.queue.fail(failed, "the frequency filter left no terms")

    answered = routes.retry()

    assert answered.json()["rows"] == 1
    assert routes.status()["rows"] == {Status.PENDING: 1}


def test_the_topics_are_counted_as_modelled_beside_the_fits(modelled) -> None:
    """One table holds both, and the status says which is which."""
    routes, _ = modelled
    routes.discover()

    rows = routes.status()["rows"]

    assert rows[Status.MODELLED] == 2, rows
    assert rows[Status.PENDING] == 1, rows


# ── Reading the topics ────────────────────────────────────────────────────


def test_the_listing_is_empty_before_a_fit(api) -> None:
    """A page has to render this."""
    routes, _ = api

    assert routes.topics() == []


def test_a_queued_fit_is_not_listed_as_a_topic(api) -> None:
    """It has no index, no language and no terms."""
    routes, _ = api
    routes.discover()

    assert routes.topics() == []


def test_the_listing_reports_each_topic_with_its_figures(modelled) -> None:
    """Everything the table on the page draws."""
    routes, _ = modelled

    listed = routes.topics()

    assert len(listed) == 2
    first = next(one for one in listed if one["topic_index"] == 0)
    assert first["language"] == "de"
    assert first["top_terms"] == ["lieferung", "versand"]
    assert first["passages"] == 1
    assert first["dominant_passages"] == 1
    assert first["documents"] == 1
    assert first["include_in_coverage"] is True
    assert first["label"] is None and first["labelled_by"] is None


def test_the_fit_state_is_empty_before_a_fit(api) -> None:
    """Nothing stored is a state the page reads, not an error."""
    routes, _ = api

    state = routes.fit()

    assert state["status"] is None
    assert state["topics"] == 0
    assert state["languages"] == []


def test_the_fit_state_carries_the_live_counts_beside_the_fitted_ones(
    modelled,
) -> None:
    """A model left stale by a re-chunk must read as stale."""
    routes, store = modelled
    store.delete_passages("de")

    state = routes.fit()

    assert state["languages"][0]["corpus_passages"] == 2
    assert state["languages"][0]["live_passages"] == 0
    assert state["languages"][0]["memberships"] == 0


def test_the_fit_state_reports_a_failure_beside_the_working_topics(modelled) -> None:
    """The topics the previous fit produced stay readable."""
    routes, store = modelled
    failed = store.request()
    store.queue.fail(failed, "no vocabulary was left")

    state = routes.fit()

    assert state["status"] == Status.FAILED
    assert "no vocabulary" in state["error"]
    assert state["topics"] == 2


def test_passages_with_no_language_are_reported(api) -> None:
    """No model covers them, and a reader has to be told."""
    routes, store = api
    store.given_passages(de=1, none=2)
    store.store(fitting("de"))

    assert routes.fit()["passages_without_language"] == 2


# ── Naming a topic ────────────────────────────────────────────────────────


def test_naming_a_topic_answers_the_topic_as_it_now_reads(modelled) -> None:
    """The page redraws from the answer rather than asking again."""
    routes, store = modelled
    topic_id = store.topic_id("de", 0)

    answered = routes.describe(topic_id, label="Shipping", include_in_coverage=True)

    assert answered.status_code == 200
    assert answered.json()["label"] == "Shipping"
    assert answered.json()["labelled_by"] == "person"


def test_taking_a_topic_out_of_coverage_is_recorded(modelled) -> None:
    """It keeps its passages either way."""
    routes, store = modelled
    topic_id = store.topic_id("de", 0)

    answered = routes.describe(topic_id, label=None, include_in_coverage=False)

    assert answered.json()["include_in_coverage"] is False
    assert answered.json()["passages"] == 1


def test_a_label_of_only_spaces_is_no_label(modelled) -> None:
    """Rather than a topic named with a blank."""
    routes, store = modelled
    topic_id = store.topic_id("de", 0)

    answered = routes.describe(topic_id, label="   ", include_in_coverage=True)

    assert answered.json()["label"] is None
    assert answered.json()["labelled_by"] is None


def test_describing_a_topic_that_does_not_exist_is_a_404(api) -> None:
    """A label is a person's, and it has to land on a topic."""
    routes, _ = api

    answered = routes.describe(999, label="Shipping", include_in_coverage=True)

    assert answered.status_code == 404
    assert answered.json()["code"] == "unknown_topic"


def test_describing_a_fit_request_is_a_404(api) -> None:
    """A fit is not a subject and is not editable."""
    routes, store = api
    fit_id = store.request()

    answered = routes.describe(fit_id, label="Nope", include_in_coverage=True)

    assert answered.status_code == 404


def test_a_label_longer_than_the_column_is_refused(modelled) -> None:
    """Refused here rather than truncated at the database."""
    routes, store = modelled
    topic_id = store.topic_id("de", 0)

    answered = routes.describe(topic_id, label="x" * 200, include_in_coverage=True)

    assert answered.status_code == 422


def test_describing_a_topic_with_no_body_at_all_takes_the_defaults(modelled) -> None:
    """Both fields are optional, and the defaults are no label, in coverage."""
    routes, store = modelled
    topic_id = store.topic_id("de", 0)

    answered = routes.describe(topic_id)

    assert answered.status_code == 200
    assert answered.json()["label"] is None
    assert answered.json()["include_in_coverage"] is True


# ── The figure ────────────────────────────────────────────────────────────


def test_a_language_no_fit_has_drawn_has_no_figure(api) -> None:
    """Produced by a fit, so before one there is nothing to serve."""
    routes, _ = api

    answered = routes.visualisation("de")

    assert answered.status_code == 404
    assert answered.json()["code"] == "no_visualisation"


@pytest.mark.parametrize("language", ["DE", "deu", "d", "1a", "d_", "de "])
def test_a_language_that_is_not_a_code_is_refused_before_the_store_is_read(
    api, language
) -> None:
    """The code becomes an object key, so it is checked first."""
    routes, _ = api

    answered = routes.visualisation(language)

    assert answered.status_code == 400, answered.status_code
    assert answered.json()["code"] == "invalid_language"


def test_a_drawn_figure_is_served_as_a_page(api, client) -> None:
    """What the Topics page embeds."""
    from api.dependencies import export_bucket

    routes, _ = api
    export_bucket.put(
        export_bucket.topic_visualisation_key("de"),
        b"<html>the de figure</html>",
        content_type=export_bucket.TOPIC_VISUALISATION_TYPE,
    )

    answered = routes.visualisation("de")

    assert answered.status_code == 200
    assert answered.content == b"<html>the de figure</html>"
    assert answered.headers["content-type"].startswith("text/html")


# ── Deleting ──────────────────────────────────────────────────────────────


def test_deleting_removes_the_topics_and_reports_what_went(modelled) -> None:
    """Passages, facts and questions stay."""
    routes, store = modelled

    answered = routes.delete()

    assert answered.status_code == 200
    body = answered.json()
    assert body["topics"] == 2
    assert body["memberships"] == 1
    assert body["languages"] == ["de"]
    assert routes.topics() == []
    assert len(list(store.queue.passages("de"))) == 2


def test_deleting_takes_the_figures_with_the_topics(modelled) -> None:
    """A figure of topics that no longer exist is worse than none."""
    from api.dependencies import export_bucket

    routes, _ = modelled
    key = export_bucket.topic_visualisation_key("de")
    export_bucket.put(
        key,
        b"<html>the de figure</html>",
        content_type=export_bucket.TOPIC_VISUALISATION_TYPE,
    )

    routes.delete()

    assert export_bucket.find(key) is None
    assert routes.visualisation("de").status_code == 404


def test_deleting_with_nothing_stored_answers_zero(api) -> None:
    """The page offers the control either way."""
    routes, _ = api

    body = routes.delete().json()

    assert (body["topics"], body["memberships"], body["labels"]) == (0, 0, 0)


def test_deleting_clears_a_queued_fit(api) -> None:
    """Otherwise a worker would fit a corpus somebody just cleared."""
    routes, _ = api
    routes.discover()

    routes.delete()

    assert routes.status()["rows"] == {}


def test_deleting_reports_the_labels_that_went_with_the_topics(modelled) -> None:
    """Nothing else stores one, so this is the warning."""
    routes, store = modelled
    routes.describe(store.topic_id("de", 0), label="Shipping")

    assert routes.delete().json()["labels"] == 1
