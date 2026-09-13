"""Fitting a topic model, and what the fit reports.

`passage_topics.weight` has a CHECK constraint of (0, 1]; gensim returns
float32, so a distribution's last member can round a hair above 1 and fail
a whole batch of memberships.
"""

from __future__ import annotations

import pytest

from topic_modelling.models import PassageVocabulary
from topic_modelling.topics import NoVocabulary, TopicFitter


def test_every_weight_is_storable(settings, passages) -> None:
    """A membership weight stays inside (0, 1]."""
    fitting = TopicFitter(**settings).fit(lambda: passages, "de")

    assert fitting.weights, "a corpus with two clear subjects produced no membership"
    for weight in fitting.weights:
        assert 0 < weight.weight <= 1, (
            f"weight {weight.weight!r} breaches the CHECK on passage_topics"
        )


def test_the_fit_reports_what_it_was_over(settings, passages) -> None:
    """The vocabulary size, the passage count and the language."""
    fitting = TopicFitter(**settings).fit(lambda: passages, "de")

    assert fitting.vocabulary > 0
    assert fitting.passages == len(passages)
    assert fitting.language == "de", "a fit belongs to the language it was over"


def test_a_corpus_that_cannot_be_rewalked_is_refused(settings, passages) -> None:
    """The fit walks the corpus several times."""
    spent = iter(passages)

    with pytest.raises(NoVocabulary):
        TopicFitter(**settings).fit(lambda: spent, "de")


def test_a_passage_holding_none_of_the_vocabulary_is_counted(
    settings, passages
) -> None:
    """It is counted once and appears in no membership."""
    counted = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=99, lemmas=["zzz", "qqq"])], "de"
    )

    assert counted.without_topics == 1, counted.without_topics
    assert not any(w.passage_id == 99 for w in counted.weights)


def test_a_passage_with_no_lemmas_at_all_is_counted(settings, passages) -> None:
    """A passage chunking stored no lemmas for is the same case."""
    empty = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=98)], "de"
    )

    assert empty.without_topics == 1, empty.without_topics


@pytest.mark.parametrize(
    "bad",
    [
        {"num_topics": 1},
        {"top_terms": 0},
        {"min_weight": 0},
        {"min_weight": 1.5},
        {"no_above": 0},
    ],
)
def test_a_fitter_that_cannot_do_its_job_is_refused(settings, bad) -> None:
    """The settings are checked when the fitter is built."""
    with pytest.raises(ValueError):
        TopicFitter(**(settings | bad))


def test_a_filter_that_removes_every_term_fails_the_run(settings, passages) -> None:
    """An empty vocabulary is a failure, not an empty fit."""
    with pytest.raises(NoVocabulary):
        TopicFitter(**(settings | {"no_below": 99})).fit(lambda: passages, "de")
