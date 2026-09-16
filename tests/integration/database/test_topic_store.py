"""The topic repositories against the database that runs them.

The unit tests prove what the fitter produces. These prove what the store
does with it: which rows a fit replaces, which it leaves, and what the
figures read back as.
"""

from __future__ import annotations

import pytest
from topics import TopicStore, fitting

from database.qa_generator import Status
from topic_modelling.models import FittedTopic

pytestmark = pytest.mark.integration


@pytest.fixture
def store(engine, database) -> TopicStore:
    """The topic tables on an empty database."""
    return TopicStore(engine)


# ── The queue ─────────────────────────────────────────────────────────────


def test_asking_for_a_fit_puts_one_request_on_the_queue(store) -> None:
    """The row is a request, not a topic: it carries no index."""
    fit_id = store.request()

    assert [row["id"] for row in store.requests()] == [fit_id]
    assert store.requests()[0]["status"] == Status.PENDING
    assert store.stored_topics() == []


def test_asking_twice_queues_one_fit(store) -> None:
    """The second ask replaces the first rather than adding to it."""
    store.request()
    second = store.request()

    assert [row["id"] for row in store.requests()] == [second]


def test_asking_again_clears_a_previous_failure(store) -> None:
    """A failed request is the asking, and the new ask replaces it."""
    failed = store.request()
    store.queue.fail(failed, "the frequency filter left no terms")

    store.request()

    assert [row["error"] for row in store.requests()] == [None]


def test_a_fit_asked_for_while_one_runs_is_queued_behind_it(store) -> None:
    """Both rows are outstanding, and only the claimed one is in progress."""
    running = store.request()
    store.claim()

    queued = store.request()

    outstanding = {row["id"]: row["status"] for row in store.requests()}
    assert outstanding == {running: Status.IN_PROGRESS, queued: Status.PENDING}


def test_a_fit_that_finishes_leaves_the_request_queued_behind_it(store) -> None:
    """The queued ask must survive the run it was made during."""
    store.given_passages(de=2)
    running = store.request()
    store.claim()
    queued = store.request()

    store.store(fitting("de"), fit_id=running)

    assert [row["id"] for row in store.requests()] == [queued]
    assert store.requests()[0]["status"] == Status.PENDING
    assert len(store.stored_topics()) == 2


def test_a_claim_takes_a_request_and_never_a_topic(store) -> None:
    """Running a topic as a fit would replace the whole table."""
    store.given_passages(de=2)
    store.store(fitting("de"))

    assert store.claim() is None, "a fitted topic was claimed as a request"


def test_stopping_withdraws_a_queued_fit(store) -> None:
    """The request is deleted rather than parked."""
    store.request()

    assert store.queue.stop() == 1
    assert store.requests() == []


def test_stopping_leaves_a_fit_already_running(store) -> None:
    """A worker holds it, and taking the row would lose its outcome."""
    store.request()
    store.claim()

    assert store.queue.stop() == 0
    assert len(store.requests()) == 1


def test_retrying_returns_a_failed_fit_to_the_queue(store) -> None:
    """The error is cleared with it."""
    failed = store.request()
    store.queue.fail(failed, "no vocabulary")

    assert store.queue.retry() == 1
    assert store.requests()[0]["status"] == Status.PENDING
    assert store.requests()[0]["error"] is None


def test_an_abandoned_claim_is_failed_by_the_next_run(store) -> None:
    """A worker that died holding a fit must not leave it invisible."""
    from datetime import timedelta

    from topic_modelling.repository import TopicQueue

    store.request()
    store.claim()

    assert TopicQueue(lease=timedelta(seconds=-1)).abandon() == 1
    assert store.requests()[0]["status"] == Status.FAILED


def test_the_languages_the_corpus_holds_come_back_commonest_first(store) -> None:
    """A fit walks them in that order."""
    store.given_passages(de=3, en=1, none=2)

    assert store.queue.languages() == ["de", "en"]


def test_a_passage_with_no_language_is_in_no_model(store) -> None:
    """No pipeline read it, so no vocabulary came off it."""
    store.given_passages(none=2)

    assert store.queue.languages() == []


def test_the_corpus_is_streamed_in_a_stable_order(store) -> None:
    """A seeded fit is only reproducible if the rows arrive the same way."""
    store.given_passages(de=4)

    first = [one.id for one in store.queue.passages("de")]
    second = [one.id for one in store.queue.passages("de")]

    assert first == sorted(first)
    assert first == second


def test_a_passage_with_no_lemmas_is_streamed_as_an_empty_vocabulary(
    store, engine
) -> None:
    """The column is nullable, and None is not a list the fitter can read."""
    from sqlalchemy import text

    store.given_passages(de=1)
    with engine.begin() as connection:
        connection.execute(text("UPDATE passages SET lemmas = NULL"))

    assert [one.lemmas for one in store.queue.passages("de")] == [[]]


def test_excerpts_come_back_for_the_passages_asked_for(store) -> None:
    """What the labeller is shown beside a topic's terms."""
    store.given_passages(de=3)

    read = store.queue.excerpts(store.passages["de"][:2])

    assert len(read) == 2
    assert all("de passage" in one for one in read), read


def test_asking_for_no_excerpt_reaches_no_database(store) -> None:
    """A topic with no strongest passage is not a query."""
    assert store.queue.excerpts([]) == []


# ── Replacing the model ───────────────────────────────────────────────────


def test_a_fit_writes_its_topics_and_their_memberships(store) -> None:
    """One transaction, every language, written `modelled` outright."""
    store.given_passages(de=2)
    weights = store.weights_for("de", (0, 0, 0.9), (1, 1, 0.8))

    written = store.store(fitting("de", weights=weights))

    assert written == 2
    assert [row["status"] for row in store.stored_topics()] == [Status.MODELLED] * 2
    assert len(store.memberships()) == 2


def test_a_membership_points_at_the_row_its_topic_became(store) -> None:
    """A topic index is not a primary key, and the two used to be confused."""
    store.given_passages(de=2)
    weights = store.weights_for("de", (0, 1, 0.7))

    store.store(fitting("de", weights=weights))

    passage_id, topic_id, _ = store.memberships()[0]
    assert passage_id == store.passages["de"][0]
    assert topic_id == store.topic_id("de", 1)


def test_two_languages_are_written_in_one_go(store) -> None:
    """One language's new topics beside another's old ones point at nothing."""
    store.given_passages(de=2, en=2)

    store.store(
        fitting("de", weights=store.weights_for("de", (0, 0, 0.9))),
        fitting("en", weights=store.weights_for("en", (0, 1, 0.6))),
    )

    assert {row["language"] for row in store.stored_topics()} == {"de", "en"}
    assert len(store.memberships()) == 2


def test_a_topic_index_identifies_a_topic_only_within_its_language(store) -> None:
    """`de #0` and `en #0` are unrelated, and both may exist."""
    store.given_passages(de=1, en=1)

    store.store(fitting("de"), fitting("en"))

    assert store.topic_id("de", 0) != store.topic_id("en", 0)


def test_a_fit_replaces_every_earlier_topic(store) -> None:
    """A refit is all-or-nothing, so nothing from the old model survives."""
    store.given_passages(de=2)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))
    was = {row["id"] for row in store.stored_topics()}

    store.store(fitting("de", weights=store.weights_for("de", (1, 0, 0.5))))

    assert not was & {row["id"] for row in store.stored_topics()}
    assert len(store.memberships()) == 1


def test_a_fit_records_what_it_was_over(store) -> None:
    """The live counts are compared against these to spot a stale model."""
    store.given_passages(de=3)

    store.store(
        fitting("de", passages=3, vocabulary=17, without_topics=1),
    )

    row = store.stored_topics()[0]
    assert (row["corpus_passages"], row["corpus_vocabulary"]) == (3, 17)
    assert row["passages_without_topics"] == 1
    assert row["fitted_at"] is not None


def test_a_fit_carries_the_time_it_was_asked_for(store) -> None:
    """A fit's provenance is when somebody wanted it, not when it ran."""
    store.given_passages(de=1)
    fit_id = store.request()
    asked = store.requests()[0]["requested_at"]

    store.store(fitting("de"), fit_id=fit_id)

    assert store.stored_topics()[0]["requested_at"] == asked


def test_a_fitted_topic_arrives_new_to_question_generation(store) -> None:
    """A refit returns every topic to that queue."""
    store.given_passages(de=1)

    store.store(fitting("de"))

    assert {row["question_status"] for row in store.stored_topics()} == {Status.NEW}


def test_a_fit_that_produced_no_topic_writes_nothing(store) -> None:
    """An empty result is not a model, and must not empty the table."""
    store.given_passages(de=1)

    assert store.store() == 0


# ── Labels ────────────────────────────────────────────────────────────────


def test_a_label_is_read_back_for_the_next_fit_to_carry(store) -> None:
    """Only the topics somebody said something about."""
    store.given_passages(de=1)
    store.store(
        fitting(
            "de",
            topics=[
                FittedTopic(0, ["lieferung"], label="Shipping", labelled_by="person"),
                FittedTopic(1, ["wartung"]),
                FittedTopic(2, ["farbe"], include_in_coverage=False),
            ],
        )
    )

    held = store.queue.labelled_topics("de")

    assert {one.topic_index for one in held} == {0, 2}
    assert next(one for one in held if one.topic_index == 0).labelled_by == "person"


def test_a_label_is_not_read_back_for_another_language(store) -> None:
    """Each language is fitted on its own vocabulary."""
    store.given_passages(de=1, en=1)
    store.store(
        fitting("de", topics=[FittedTopic(0, ["lieferung"], label="Shipping")]),
        fitting("en", topics=[FittedTopic(0, ["delivery"])]),
    )

    assert store.queue.labelled_topics("en") == []


def test_naming_a_topic_records_that_a_person_did(store) -> None:
    """Provenance, so a report can say which names it is reading."""
    store.given_passages(de=1)
    store.store(fitting("de"))
    topic_id = store.topic_id("de", 0)

    described = store.catalog.describe(
        topic_id, label="  Shipping  ", include_in_coverage=True
    )

    assert described is not None
    assert described.label == "Shipping", described.label
    assert described.labelled_by == "person"


def test_clearing_a_label_clears_what_named_it(store) -> None:
    """A topic with no name has nothing that named it."""
    store.given_passages(de=1)
    store.store(
        fitting("de", topics=[FittedTopic(0, ["a"], label="Was", labelled_by="person")])
    )
    topic_id = store.topic_id("de", 0)

    described = store.catalog.describe(topic_id, label="   ", include_in_coverage=True)

    assert described is not None
    assert (described.label, described.labelled_by) == (None, None)


def test_taking_a_topic_out_of_coverage_leaves_its_passages(store) -> None:
    """It is a reporting decision, not a deletion."""
    store.given_passages(de=2)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))
    topic_id = store.topic_id("de", 0)

    described = store.catalog.describe(topic_id, label=None, include_in_coverage=False)

    assert described is not None
    assert described.include_in_coverage is False
    assert described.passages == 1, described.passages


def test_a_fit_request_is_not_a_topic_and_cannot_be_named(store) -> None:
    """Both live in `topics`, and only one of them is a subject."""
    fit_id = store.request()

    assert (
        store.catalog.describe(fit_id, label="Nope", include_in_coverage=True) is None
    )


def test_naming_a_topic_that_does_not_exist_answers_nothing(store) -> None:
    """Rather than writing a row nobody asked for."""
    assert store.catalog.describe(999, label="Nope", include_in_coverage=True) is None


# ── Reading the model back ────────────────────────────────────────────────


def test_a_topic_reports_how_much_of_the_corpus_it_holds(store) -> None:
    """Every passage holding it, and the mean weight over those."""
    store.given_passages(de=3)
    store.store(
        fitting(
            "de",
            weights=store.weights_for("de", (0, 0, 0.8), (1, 0, 0.6), (2, 1, 0.9)),
        )
    )

    first = next(one for one in store.catalog.topics() if one.topic_index == 0)

    assert first.passages == 2, first.passages
    assert abs(first.mean_weight - 0.7) < 1e-6, first.mean_weight
    assert first.documents == 1, first.documents


def test_a_topic_counts_only_the_passages_it_is_strongest_in_as_its_own(
    store,
) -> None:
    """A passage belongs to several topics, so counting all dilutes each."""
    store.given_passages(de=2)
    store.store(
        fitting(
            "de",
            weights=store.weights_for("de", (0, 0, 0.9), (0, 1, 0.1), (1, 1, 0.7)),
        )
    )

    topics = {one.topic_index: one for one in store.catalog.topics()}

    assert topics[0].passages == 1 and topics[0].dominant_passages == 1
    assert topics[1].passages == 2 and topics[1].dominant_passages == 1


def test_a_topic_holding_nothing_reports_zero_rather_than_nothing(store) -> None:
    """A topic with no membership is still a topic and must still list."""
    store.given_passages(de=1)
    store.store(fitting("de"))

    topics = store.catalog.topics()

    assert len(topics) == 2
    assert all(one.passages == 0 for one in topics)
    assert all(one.mean_weight == 0.0 for one in topics)
    assert all(one.dominant_passages == 0 for one in topics)


def test_a_topic_reports_how_many_of_its_passages_are_tables(store) -> None:
    """A topic that is mostly tables is a document's apparatus, not a subject."""
    store.given_passages(de=1)
    table = store.given_table_passage("de")
    held = store.passages["de"]
    store.store(
        fitting(
            "de",
            weights=store.weights_for(
                "de", (held.index(held[0]), 0, 0.9), (held.index(table), 0, 0.8)
            ),
        )
    )

    first = next(one for one in store.catalog.topics() if one.topic_index == 0)

    assert first.dominant_passages == 2, first.dominant_passages
    assert first.table_passages == 1, first.table_passages


def test_a_topic_reports_the_validated_facts_drawn_from_its_passages(store) -> None:
    """A question can only be asked from a validated fact."""
    store.given_passages(de=2)
    store.given_facts(store.passages["de"][0], validated=2, rejected=3)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))

    first = next(one for one in store.catalog.topics() if one.topic_index == 0)

    assert first.validated_facts == 2, first.validated_facts


def test_the_topics_come_back_ordered_by_language_then_index(store) -> None:
    """The table on the page is drawn in this order."""
    store.given_passages(de=1, en=1)
    store.store(fitting("de"), fitting("en"))

    read = [(one.language, one.topic_index) for one in store.catalog.topics()]

    assert read == sorted(read)


def test_a_fit_request_is_not_listed_as_a_topic(store) -> None:
    """It has no index, no language and no terms to show."""
    store.request()

    assert store.catalog.topics() == []


# ── The state of the model ────────────────────────────────────────────────


def test_nothing_stored_reports_no_model_at_all(store) -> None:
    """A new deployment, which the page has to render."""
    state = store.catalog.fit_state()

    assert state.status is None
    assert state.topics == 0
    assert state.languages == []
    assert state.stale is False


def test_a_fresh_fit_is_not_stale(store) -> None:
    """The passages it was fitted over are the passages the corpus holds."""
    store.given_passages(de=2)
    store.store(fitting("de", passages=2, weights=store.weights_for("de", (0, 0, 0.9))))

    state = store.catalog.fit_state()

    assert state.status == "modelled"
    assert state.topics == 2
    assert [one.language for one in state.languages] == ["de"]
    assert state.stale is False, state.languages


def test_a_rechunk_that_deleted_the_passages_leaves_the_model_stale(store) -> None:
    """The memberships went with them, so nothing is topic-weighted now."""
    store.given_passages(de=2)
    store.store(fitting("de", passages=2, weights=store.weights_for("de", (0, 0, 0.9))))

    store.delete_passages("de")

    state = store.catalog.fit_state()
    assert state.stale is True
    assert state.languages[0].memberships == 0
    assert state.languages[0].live_passages == 0


def test_a_corpus_that_has_grown_leaves_the_model_stale(store, engine) -> None:
    """Adding a document makes every topic stale, not one."""
    from seed import digest, document, passage
    from sqlalchemy.orm import Session

    store.given_passages(de=2)
    store.store(fitting("de", passages=2, weights=store.weights_for("de", (0, 0, 0.9))))
    with Session(engine) as session:
        session.add(document(digest("added")))
        session.flush()
        session.add(passage(digest("added"), ordinal=1, language="de", lemmas=["neu"]))
        session.commit()

    state = store.catalog.fit_state()

    assert state.languages[0].live_passages == 3
    assert state.languages[0].corpus_passages == 2
    assert state.stale is True


def test_passages_carrying_no_language_are_reported_separately(store) -> None:
    """No model covers them, whatever the fit did."""
    store.given_passages(de=1, none=3)
    store.store(fitting("de"))

    assert store.catalog.fit_state().passages_without_language == 3


def test_a_failed_fit_is_reported_beside_the_topics_it_did_not_replace(store) -> None:
    """The reason goes on the request row; the working topics stay readable."""
    store.given_passages(de=1)
    store.store(fitting("de"))
    failed = store.request()
    store.queue.fail(failed, "the frequency filter left no terms")

    state = store.catalog.fit_state()

    assert state.status == Status.FAILED
    assert state.error is not None and "frequency" in state.error
    assert state.topics == 2, "the previous topics went with the failure"


# ── Counting and deleting ─────────────────────────────────────────────────


def test_the_counts_separate_the_topics_from_the_fits(store) -> None:
    """`modelled` counts topics; the rest describe a fit."""
    store.given_passages(de=2)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))
    store.request()

    counts = store.queue.counts()

    assert counts[Status.MODELLED] == 2
    assert counts[Status.PENDING] == 1
    assert counts["memberships"] == 1
    assert counts["passages_with_a_topic"] == 1


def test_a_passage_in_two_topics_is_counted_once_as_covered(store) -> None:
    """Memberships overlap; coverage does not."""
    store.given_passages(de=1)
    store.store(
        fitting("de", weights=store.weights_for("de", (0, 0, 0.6), (0, 1, 0.4)))
    )

    counts = store.queue.counts()

    assert counts["memberships"] == 2
    assert counts["passages_with_a_topic"] == 1


def test_deleting_removes_every_topic_and_says_what_went(store) -> None:
    """Including the languages, so the figures keyed by one can go too."""
    store.given_passages(de=1, en=1)
    store.store(
        fitting("de", topics=[FittedTopic(0, ["a"], label="Named")]),
        fitting(
            "en",
            topics=[FittedTopic(0, ["b"])],
            weights=store.weights_for("en", (0, 0, 0.9)),
        ),
    )

    removed = store.catalog.delete_all()

    assert removed.topics == 2
    assert removed.memberships == 1
    assert removed.labels == 1
    assert sorted(removed.languages) == ["de", "en"]
    assert store.rows() == []


def test_deleting_takes_a_queued_fit_with_it(store) -> None:
    """Otherwise a worker would fit a corpus somebody just cleared."""
    store.request()

    store.catalog.delete_all()

    assert store.rows() == []


def test_deleting_leaves_the_passages_alone(store) -> None:
    """A topic is a reading of the corpus, not part of it."""
    store.given_passages(de=2)
    store.store(fitting("de", weights=store.weights_for("de", (0, 0, 0.9))))

    store.catalog.delete_all()

    assert len(list(store.queue.passages("de"))) == 2
