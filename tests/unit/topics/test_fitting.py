"""Fitting a topic model, and what the fit reports."""

from __future__ import annotations

import pytest
from topic_drivers import FitterDriver, corpus

from topic_modelling.models import PassageVocabulary
from topic_modelling.topics import NoVocabulary


def test_every_weight_is_storable(fitter) -> None:
    """A membership weight stays inside (0, 1], as the column's CHECK needs."""
    fitting = fitter.fit("de")

    assert fitting.fitting.weights, "two clear subjects produced no membership"
    for weight in fitting.fitting.weights:
        assert 0 < weight.weight <= 1, (
            f"weight {weight.weight!r} breaches the CHECK on passage_topics"
        )


def test_the_fit_reports_what_it_was_over(fitter, passages) -> None:
    """The vocabulary size, the passage count and the language."""
    fitting = fitter.fit("de").fitting

    assert fitting.vocabulary > 0
    assert fitting.passages == len(passages)
    assert fitting.language == "de", "a fit belongs to the language it was over"


def test_the_fit_produces_the_number_of_topics_it_was_asked_for(fitter) -> None:
    """TOPIC_NUM_TOPICS is a count, not a ceiling."""
    assert fitter.fit("de").topic_count == 2


def test_every_topic_carries_a_signature_no_longer_than_asked_for(
    settings, passages
) -> None:
    """TOPIC_TOP_TERMS caps it; a small vocabulary may not reach it."""
    fitting = FitterDriver(settings | {"top_terms": 3}, passages).fit("de")

    for index in range(fitting.topic_count):
        terms = fitting.terms_of(index)
        assert 0 < len(terms) <= 3, terms
        assert len(set(terms)) == len(terms), f"{terms} repeats a term"


def test_the_same_seed_gives_the_same_model(settings, passages) -> None:
    """A reference dataset's topic weighting must not move between runs."""
    first = FitterDriver(settings, passages).fit("de")
    second = FitterDriver(settings, passages).fit("de")

    assert [first.terms_of(i) for i in range(2)] == [
        second.terms_of(i) for i in range(2)
    ]
    assert first.fitting.weights == second.fitting.weights


def test_the_corpus_is_walked_more_than_once(fitter) -> None:
    """Once to build the vocabulary, once per pass, once to score."""
    fitter.fit("de")

    assert fitter.walks > 2, fitter.walks


def test_a_corpus_that_cannot_be_rewalked_is_refused(fitter) -> None:
    """A spent generator would fit a model over nothing and report success."""
    with pytest.raises(NoVocabulary, match="second walk"):
        fitter.fit_once_walkable("de")


def test_an_empty_corpus_is_refused_by_name(settings) -> None:
    """The message has to say which language had nothing."""
    with pytest.raises(NoVocabulary, match="no de passages"):
        FitterDriver(settings).fit("de")


def test_a_filter_that_removes_every_term_fails_the_run(settings, passages) -> None:
    """An empty vocabulary is a failure, not an empty fit."""
    with pytest.raises(NoVocabulary, match="frequency filter"):
        FitterDriver(settings | {"no_below": 99}, passages).fit("de")


def test_the_refusal_says_what_the_filter_was_and_what_it_left(
    settings, passages
) -> None:
    """A reader has to be able to tell which setting to move."""
    with pytest.raises(NoVocabulary) as raised:
        FitterDriver(settings | {"no_below": 99}, passages).fit("de")

    said = str(raised.value)
    assert "99" in said, said
    assert "4 passage" in said, said


# ── What falls out of a topic ─────────────────────────────────────────────


def test_a_passage_holding_none_of_the_vocabulary_is_counted(fitter) -> None:
    """It is counted once and appears in no membership."""
    counted = fitter.with_passage(99, "zzz", "qqq").fit("de")

    assert counted.unplaced == 1, counted.unplaced
    assert 99 not in counted.placed_ids


def test_a_passage_with_no_lemmas_at_all_is_counted(fitter) -> None:
    """A passage chunking stored no lemmas for is the same case."""
    empty = fitter.holding(PassageVocabulary(id=98)).fit("de")

    assert empty.unplaced == 1, empty.unplaced


def test_a_weight_floor_above_every_weight_places_nothing(settings, passages) -> None:
    """The floor is the other reason a passage can end up unplaced."""
    floored = FitterDriver(settings | {"min_weight": 1.0}, passages).fit("de")

    assert floored.fitting.weights == []
    assert floored.unplaced == len(passages), floored.unplaced


def test_the_weight_floor_is_exclusive(settings, passages) -> None:
    """A weight exactly at the floor is not above it, so it is not stored."""
    fitting = FitterDriver(settings, passages).fit("de")
    lowest = min(weight.weight for weight in fitting.fitting.weights)

    at_the_floor = FitterDriver(settings | {"min_weight": lowest}, passages).fit("de")

    assert lowest not in [w.weight for w in at_the_floor.fitting.weights]


def test_a_term_in_every_passage_is_dropped_from_the_vocabulary(
    settings, passages
) -> None:
    """The boilerplate filter, which no word list could know about."""
    everywhere = [
        PassageVocabulary(id=one.id, lemmas=[*one.lemmas, "seitenfuss"])
        for one in passages
    ]

    fitting = FitterDriver(settings, everywhere).fit("de")

    assert "seitenfuss" not in fitting.space.vocabulary, fitting.space.vocabulary


def test_a_term_in_too_few_passages_is_dropped_from_the_vocabulary(fitter) -> None:
    """The typo and one-off product name filter."""
    fitting = fitter.with_passage(97, "lieferung", "einmaligerbegriff").fit("de")

    assert "einmaligerbegriff" not in fitting.space.vocabulary


def test_two_clear_subjects_come_out_as_two_topics(fitter, passages) -> None:
    """The passages about one subject do not land on the topic of the other."""
    fitting = fitter.fit("de")

    shipping = {fitting.dominant_topic(1), fitting.dominant_topic(2)}
    maintenance = {fitting.dominant_topic(3), fitting.dominant_topic(4)}

    assert len(shipping) == 1, f"the two shipping passages split: {shipping}"
    assert len(maintenance) == 1, f"the two maintenance passages split: {maintenance}"
    assert shipping != maintenance, "both subjects landed on one topic"


def test_a_topic_signature_is_drawn_from_the_terms_of_its_own_subject(fitter) -> None:
    """A signature mixing both subjects is a model that has not separated."""
    fitting = fitter.fit("de")
    shipping = fitting.terms_of(fitting.dominant_topic(1) or 0)

    assert "lieferung" in shipping, shipping
    assert "wartung" not in shipping, shipping


# ── Settings the fitter refuses ───────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        {"num_topics": 1},
        {"num_topics": 0},
        {"num_topics": -1},
        {"top_terms": 0},
        {"top_terms": -1},
        {"min_weight": 0},
        {"min_weight": -0.1},
        {"min_weight": 1.5},
        {"no_above": 0},
        {"no_above": -0.5},
        {"no_above": 1.01},
    ],
)
def test_a_fitter_that_cannot_do_its_job_is_refused(settings, bad) -> None:
    """The settings are checked when the fitter is built, not at the fit."""
    with pytest.raises(ValueError):
        FitterDriver(settings | bad)


@pytest.mark.parametrize(
    "allowed", [{"min_weight": 1}, {"no_above": 1}, {"num_topics": 2}, {"top_terms": 1}]
)
def test_the_edge_of_each_range_is_allowed(settings, allowed) -> None:
    """The bounds are inclusive where the message says they are."""
    FitterDriver(settings | allowed)


def test_a_settings_refusal_names_the_environment_variable(settings) -> None:
    """A reader has to know which line of the env file to change."""
    with pytest.raises(ValueError, match="TOPIC_NUM_TOPICS"):
        FitterDriver(settings | {"num_topics": 1})


def test_a_corpus_of_one_passage_has_no_vocabulary_to_tell_apart(settings) -> None:
    """Every term appears in the only passage, so the share filter takes it."""
    with pytest.raises(NoVocabulary):
        FitterDriver(settings, corpus("lieferung versand transport")).fit("de")
