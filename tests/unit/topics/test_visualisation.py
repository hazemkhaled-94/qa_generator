"""The figure drawn of a fitted model."""

from __future__ import annotations

import json
import re

import pytest
from topic_drivers import FitterDriver

from topic_modelling.models import TopicSpace
from topic_modelling.visualisation import _TERMS_SHOWN, render

#: The payload pyLDAvis embeds in the page it writes.
_PAYLOAD = re.compile(r"var \w+_data = (\{.*?\});", re.DOTALL)


def payload_of(page: str) -> dict:
    """Reads the model back out of the page that draws it."""
    found = _PAYLOAD.search(page)
    assert found, "the page carries no embedded model"
    return json.loads(found.group(1))


# ── The space a fit hands over ────────────────────────────────────────────


def test_the_term_matrix_matches_the_fit(fitting) -> None:
    """One row per topic, each a distribution over the whole vocabulary."""
    space = fitting.space

    assert len(space.topic_term) == 2, len(space.topic_term)
    assert len(space.vocabulary) == fitting.fitting.vocabulary
    assert len(space.term_frequency) == fitting.fitting.vocabulary
    for row in space.topic_term:
        assert len(row) == fitting.fitting.vocabulary, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)


def test_the_document_matrix_holds_one_row_per_scored_passage(fitting) -> None:
    """Each row is a distribution, and each document has a length."""
    space = fitting.space

    assert len(space.doc_lengths) == len(space.doc_topic)
    for row in space.doc_topic:
        assert len(row) == 2, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)
    assert all(length > 0 for length in space.doc_lengths), space.doc_lengths


def test_a_document_length_counts_occurrences_not_distinct_terms(settings) -> None:
    """PyLDAvis weights a term by how often it appears, so repeats count."""
    from topic_drivers import corpus

    repeated = FitterDriver(
        settings,
        corpus(
            "lieferung lieferung lieferung versand",
            "lieferung versand versand transport",
            "wartung reparatur reparatur ersatzteil",
            "wartung wartung reparatur pruefung",
        ),
    ).fit("de")

    assert max(repeated.space.doc_lengths) >= 4, repeated.space.doc_lengths


def test_every_term_frequency_is_positive(fitting) -> None:
    """A term in the vocabulary appeared, or the filter would have taken it."""
    assert all(count > 0 for count in fitting.space.term_frequency)


def test_a_passage_holding_none_of_the_vocabulary_is_in_no_matrix(fitter) -> None:
    """It is in no membership and no figure, and counted once."""
    lost = fitter.with_passage(99, "zzz", "qqq").fit("de")

    assert lost.unplaced == 1, lost.unplaced
    assert len(lost.space.doc_topic) == 4, len(lost.space.doc_topic)


# ── The page ──────────────────────────────────────────────────────────────


def test_a_topic_keeps_its_number_in_the_figure(fitting) -> None:
    """Topic n in the figure is topic n in the topics table."""
    page = fitting.drawn()

    assert '"topic.order": [0, 1]' in page, re.findall(r'"topic\.order":[^]]*]', page)


def test_the_page_reaches_no_network(fitting) -> None:
    """Nothing is fetched when the page is opened."""
    page = fitting.drawn()

    assert not fitting.fetched_by(page), fitting.fetched_by(page)
    assert "var LDAvis" in page, "the LDAvis script is not inlined"
    assert "d3.select" in page, "d3 is not inlined"


def test_the_stylesheet_is_inlined_rather_than_linked(fitting) -> None:
    """The template would otherwise link one from a CDN."""
    page = fitting.drawn()

    assert page.startswith("<style>"), page[:80]
    assert "LDAvis" in page[: page.index("</style>")], "it is not the LDAvis CSS"
    assert 'ldavis_css_url="http' not in page, "a stylesheet is still linked"


def test_the_page_is_utf8(fitting) -> None:
    """A German term carries characters ASCII has no room for."""
    drawn = render(fitting.space, "de")

    assert isinstance(drawn, bytes)
    assert drawn.decode("utf-8"), "the page is not valid UTF-8"


def test_the_embedded_model_holds_real_numbers(fitting) -> None:
    """A complex coordinate is one json refuses to write at all."""
    placed = payload_of(fitting.drawn())["mdsDat"]

    assert len(placed["x"]) == 2, placed
    for axis in ("x", "y"):
        assert all(isinstance(one, (int, float)) for one in placed[axis]), placed[axis]


def test_the_term_list_is_capped_by_the_vocabulary(fitting) -> None:
    """A vocabulary smaller than the bar chart must not be over-asked."""
    payload = payload_of(fitting.drawn())

    assert payload["R"] == min(_TERMS_SHOWN, len(fitting.space.vocabulary))
    assert payload["R"] <= len(fitting.space.vocabulary), payload["R"]


def test_drawing_the_same_model_twice_gives_the_same_page(fitting) -> None:
    """The inlined assets are cached, and a cache must not leak state."""
    first = render(fitting.space, "de")
    second = render(fitting.space, "de")

    assert payload_of(first.decode()) == payload_of(second.decode())


def test_a_model_that_placed_no_passage_is_not_drawn(fitting) -> None:
    """There is nothing to draw, and an empty figure would read as a model."""
    space = fitting.space

    with pytest.raises(ValueError, match="nothing to draw"):
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


def test_the_refusal_names_the_language(fitting) -> None:
    """A run fits several, so the reason has to say which one this was."""
    with pytest.raises(ValueError, match="no en passage"):
        render(
            TopicSpace(
                topic_term=fitting.space.topic_term,
                doc_topic=[],
                doc_lengths=[],
                vocabulary=fitting.space.vocabulary,
                term_frequency=fitting.space.term_frequency,
            ),
            "en",
        )
