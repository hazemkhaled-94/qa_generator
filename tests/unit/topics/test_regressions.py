"""Defects this service has had, one test each.

Every test here failed once. The name says the symptom rather than the
mechanism, so a reader who sees one go red knows what a user would see; the
docstring says what went wrong underneath.
"""

from __future__ import annotations

import json
import re
from math import isnan

import pytest
from topic_drivers import FitterDriver, LabellerDriver, corpus

from topic_modelling.models import FittedTopic, PassageVocabulary, TopicSpace
from topic_modelling.topics import NoVocabulary, carry_labels
from topic_modelling.visualisation import render


def test_a_whole_batch_of_memberships_is_not_lost_to_a_rounding_error(
    fitter,
) -> None:
    """Every stored weight sits inside the column's CHECK.

    Gensim returns float32, and a distribution's last member rounded to
    1.0000001, which breached the CHECK and rolled back every membership of
    the batch it was in.
    """
    for weight in fitter.fit("de").fitting.weights:
        assert weight.weight <= 1.0, repr(weight.weight)
        assert weight.weight > 0.0, repr(weight.weight)


def test_a_spent_generator_is_not_fitted_over_nothing(fitter) -> None:
    """A corpus that cannot be re-read is refused.

    Handed one as a generator, the vocabulary pass exhausted it, so the
    factorisation saw an empty corpus and the run reported success.
    """
    with pytest.raises(NoVocabulary):
        fitter.fit_once_walkable("de")


def test_the_figure_numbers_its_topics_as_the_table_does(fitting) -> None:
    """Topic n in the figure is topic n in the table.

    PyLDAvis sorts topics by prevalence unless told not to, so the two
    disagreed about which topic was which.
    """
    assert '"topic.order": [0, 1]' in fitting.drawn()


def test_the_figure_can_be_written_at_all(fitting) -> None:
    """The page carries a model json can parse.

    PyLDAvis eigendecomposes a symmetric matrix with `eig`, which returns
    complex values, and json.dumps then refused to write the page.
    """
    page = fitting.drawn()
    found = re.search(r"var \w+_data = (\{.*?\});", page, re.DOTALL)

    assert found, "no model was embedded"
    assert json.loads(found.group(1)), "the embedded model is not JSON"


def test_a_closing_quote_is_not_left_on_a_name() -> None:
    """A name comes out of the model with no wrapping on it.

    Stripping the full stop and then the quote left `Anti-money laundering"`
    when the model wrote `"Anti-money laundering."`.
    """
    assert LabellerDriver('"Anti-money laundering."').names("a") == (
        "Anti-money laundering"
    )


def test_a_name_a_person_typed_is_not_replaced_by_a_models(service) -> None:
    """A hand-typed label survives the refit that carried it over.

    A refit deletes every topic, and the labeller then renamed the topic the
    label had just been carried onto.
    """
    named = LabellerDriver("Generated", languages={"de": "German"})
    flow = service(labeller=named)
    flow.queue.previous["de"] = [
        FittedTopic(
            0,
            ["lieferung", "versand", "transport", "monatlich", "paketdienst"],
            label="By hand",
            labelled_by="person",
        )
    ]

    flow.request_and_run()

    assert "By hand" in flow.labels_of("de"), flow.labels_of("de")


def test_one_label_does_not_land_on_two_topics() -> None:
    """A previous label is used at most once.

    Every topic took its best match independently, so a refit that split one
    subject in two put the same name on both halves.
    """
    terms = ["lieferung", "versand", "transport", "paket", "monatlich"]
    twice = carry_labels(
        [FittedTopic(0, terms), FittedTopic(1, terms)],
        [FittedTopic(0, terms, label="Shipping")],
    )

    assert [one.label for one in twice].count("Shipping") == 1, twice


def test_a_fit_queued_while_one_runs_is_not_swallowed(service) -> None:
    """A second discover during a fit still runs.

    `request` deleted every request row including the claimed one, and the
    fit in progress then deleted the request queued behind it, so the second
    ask did nothing at all. Pinned here for the shape of the flow; the SQL
    that lost the row is pinned in tests/integration.
    """
    flow = service()
    flow.request()
    flow.run()
    flow.request()

    assert flow.queue.pending, "the second request was lost"
    assert flow.drain() == 1, "the queued fit never ran"


def test_a_passage_the_weight_floor_emptied_is_counted_as_unplaced(
    settings, passages
) -> None:
    """Both reasons a passage can hold no topic are counted.

    Only a passage holding none of the vocabulary used to be counted, so a
    floor that rejected every weight reported full coverage.
    """
    floored = FitterDriver(settings | {"min_weight": 1.0}, passages).fit("de")

    assert floored.unplaced == len(passages), floored.unplaced
    assert floored.fitting.weights == []


def test_a_passage_with_no_lemmas_does_not_divide_by_zero(fitter) -> None:
    """A passage holding none of the vocabulary is skipped, not normalised.

    Normalising a passage's distribution divided by its own sum, which is
    zero for such a passage.
    """
    empty = fitter.holding(PassageVocabulary(id=98)).fit("de")

    assert empty.unplaced == 1
    assert all(abs(sum(row) - 1) < 1e-6 for row in empty.space.doc_topic)


def test_a_corpus_tfidf_weighs_flat_is_refused_rather_than_fitted(settings) -> None:
    """A model of NaN is refused instead of being stored as topics.

    A term in every passage has an idf of zero. With TOPIC_NO_ABOVE at 1 the
    filter keeps such terms, and a corpus where every surviving term is one
    weighed to an all-zero matrix: the factorisation divided by its norm,
    every topic came out NaN, and the fit reported success with two
    identically signed topics, no memberships and a NaN figure.
    """
    flat = FitterDriver(
        settings | {"no_below": 1, "no_above": 1.0},
        corpus("lieferung versand", "lieferung versand"),
    )

    with pytest.raises(NoVocabulary, match="TOPIC_NO_ABOVE"):
        flat.fit("de")


def test_no_topic_weight_is_ever_nan(fitter) -> None:
    """A NaN weight compares false against the floor, so it vanished silently."""
    fitting = fitter.fit("de")

    for row in fitting.space.topic_term:
        assert not any(isnan(value) for value in row), row
    assert not any(isnan(one.weight) for one in fitting.fitting.weights)


def test_an_empty_model_is_refused_rather_than_drawn(fitting) -> None:
    """A model that placed no passage raises instead of drawing.

    PyLDAvis was handed an empty document matrix and wrote a page showing a
    model with no topics in it, which read as a successful fit.
    """
    with pytest.raises(ValueError):
        render(
            TopicSpace(
                topic_term=fitting.space.topic_term,
                doc_topic=[],
                doc_lengths=[],
                vocabulary=fitting.space.vocabulary,
                term_frequency=fitting.space.term_frequency,
            ),
            "de",
        )


def test_a_figure_that_cannot_be_drawn_does_not_lose_the_topics(service) -> None:
    """A rendering failure leaves the topics written.

    The figure was drawn inside the transaction that wrote them, so a
    failure there rolled the whole fit back.
    """
    flow = service()
    flow.export.refuses_put = RuntimeError("the bucket is full")

    flow.request_and_run()

    assert flow.languages_stored == ["de"], flow.languages_stored
    assert flow.failure is None, flow.failure


def test_one_languages_failure_does_not_take_the_others_topics(
    service, passages
) -> None:
    """A language too small to fit is left out, not fatal.

    It raised, and the run failed without writing the languages that had
    fitted perfectly well.
    """
    flow = service({"de": passages, "en": corpus("solitary", first_id=900)})

    flow.request_and_run()

    assert flow.languages_stored == ["de"], flow.languages_stored
    assert flow.failure is None, flow.failure
