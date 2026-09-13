"""The figure drawn of a fitted model.

pyLDAvis reorders topics by prevalence unless told not to, which would number
the figure differently from the topics table beside it.
"""

from __future__ import annotations

import re

import pytest

from topic_modelling.models import PassageVocabulary, TopicSpace
from topic_modelling.topics import TopicFitter
from topic_modelling.visualisation import render


@pytest.fixture
def fitting(settings, passages):
    """One fitted model over the shared corpus."""
    return TopicFitter(**settings).fit(lambda: passages, "de")


def test_the_term_matrix_matches_the_fit(fitting) -> None:
    """One row per topic, each a distribution over the whole vocabulary."""
    space = fitting.space

    assert len(space.topic_term) == 2, len(space.topic_term)
    assert len(space.vocabulary) == fitting.vocabulary, len(space.vocabulary)
    assert len(space.term_frequency) == fitting.vocabulary
    for row in space.topic_term:
        assert len(row) == fitting.vocabulary, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)


def test_the_document_matrix_holds_one_row_per_placed_passage(
    fitting, passages
) -> None:
    """Each row is a distribution, and each document has a length."""
    space = fitting.space

    assert len(space.doc_topic) == len(passages) - fitting.without_topics
    assert len(space.doc_lengths) == len(space.doc_topic)
    for row in space.doc_topic:
        assert len(row) == 2, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)
    assert all(length > 0 for length in space.doc_lengths), space.doc_lengths


def test_a_passage_holding_none_of_the_vocabulary_is_in_neither(
    settings, passages
) -> None:
    """It is in no membership and no figure, and counted once."""
    lost = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=99, lemmas=["zzz", "qqq"])], "de"
    )

    assert lost.without_topics == 1, lost.without_topics
    assert len(lost.space.doc_topic) == len(passages), len(lost.space.doc_topic)


def test_a_topic_keeps_its_number_in_the_figure(fitting) -> None:
    """Topic n in the figure is topic n in the topics table."""
    page = render(fitting.space, "de").decode("utf-8")

    assert '"topic.order": [0, 1]' in page, re.findall(r'"topic\.order":[^]]*]', page)


def test_the_page_reaches_no_network(fitting) -> None:
    """Nothing is fetched when the page is opened."""
    page = render(fitting.space, "de").decode("utf-8")

    fetched = [
        url
        for url in re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', page)
        if url.startswith(("http://", "https://", "//"))
    ]
    assert not fetched, fetched
    assert "var LDAvis" in page, "the LDAvis script is not inlined"
    assert "d3.select" in page, "d3 is not inlined"


def test_a_model_that_placed_no_passage_is_not_drawn(fitting) -> None:
    """There is nothing to draw."""
    space = fitting.space

    with pytest.raises(ValueError):
        render(
            TopicSpace(
                topic_term=space.topic_term,
                doc_topic=[],
                doc_lengths=[],
                vocabulary=space.vocabulary,
                term_frequency=space.term_frequency,
            ),
            "de",
        )
