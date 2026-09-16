"""The linguistic surface chunking writes onto every passage.

The language, the numbered units a fact cites and the vocabulary a topic
model is fitted over are all read here, with the real spaCy pipelines.
"""

from __future__ import annotations

import pytest

from preprocessing.chunking.models import Chunk
from preprocessing.chunking.passages import PassageBuilder

pytestmark = pytest.mark.nlp

GERMAN = (
    "Die Bundesanstalt für Finanzdienstleistungsaufsicht beaufsichtigt die "
    "Institute. Sie veröffentlicht ihre Feststellungen jedes Jahr."
)
ENGLISH = (
    "The supervisory authority oversees the institutions. It publishes its "
    "findings every year."
)

#: A table as the Markdown serializer renders one.
RENDERED = "| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"


def chunk(text: str, *, tables: bool = False) -> Chunk:
    """One passage as the chunk reader produced it, before it is read."""
    return Chunk(
        ordinal=1,
        text=text,
        page_from=None,
        page_to=None,
        section_path=None,
        block_type="table" if tables else "text",
        table_cells=[{"cells": []}] if tables else [],
    )


def read(*passages: Chunk, language: str | None = "en") -> list[Chunk]:
    """Reads the surface of some passages the way the builder does."""
    return PassageBuilder._read(list(passages), language)


def test_each_passage_is_read_under_the_language_of_its_own_text() -> None:
    """One file carries a German report and its English summary."""
    read_back = read(chunk(GERMAN), chunk(ENGLISH), language="de")

    assert [one.language for one in read_back] == ["de", "en"]


def test_a_passage_too_short_to_judge_keeps_the_documents_language() -> None:
    """Below the detector's floor it is guessing from a few words."""
    assert read(chunk("Anhang"), language="de")[0].language == "de"


def test_a_passage_of_a_document_with_no_language_is_read_as_none() -> None:
    """Which is what puts it outside every topic model."""
    assert read(chunk("Anhang"), language=None)[0].language is None


def test_a_prose_passage_is_numbered_by_its_sentences() -> None:
    """Each one slicing to itself out of the passage text."""
    numbered = read(chunk(ENGLISH))[0].sentences

    assert [one["i"] for one in numbered] == [0, 1]
    for unit in numbered:
        sliced = ENGLISH[unit["start"] : unit["end"]]
        assert sliced.strip() == sliced and sliced, repr(sliced)


def test_a_sentence_carrying_a_claim_is_counted_as_one() -> None:
    """A sentence with several should yield several facts, not one."""
    numbered = read(chunk(ENGLISH))[0].sentences

    assert [one["predicates"] for one in numbered] == [1, 1]


def test_a_german_obligation_is_read_as_a_claim() -> None:
    """The medium pipeline tags a modal as finite; the small one does not."""
    sentence = "Ein Risikobericht muss jährlich erstellt werden."

    numbered = read(chunk(sentence), language="de")[0].sentences

    assert numbered[0]["predicates"] >= 1, numbered


def test_a_table_passage_is_numbered_by_its_rendered_rows() -> None:
    """A table has no sentences, so its rows are what a fact cites."""
    numbered = read(chunk(RENDERED, tables=True))[0].sentences

    assert [one["i"] for one in numbered] == [0, 1, 2]
    assert RENDERED[numbered[2]["start"] : numbered[2]["end"]] == "| Compact | 4 |"


def test_a_table_passage_is_still_read_for_vocabulary() -> None:
    """Its headings and cell values are what it is about."""
    terms = read(chunk(RENDERED, tables=True))[0].lemmas

    assert "model" in terms, terms
    assert "mass" in terms


def test_the_vocabulary_is_the_content_words_and_nothing_else() -> None:
    """Nouns, proper nouns and adjectives, lemmatised."""
    terms = read(chunk(ENGLISH))[0].lemmas

    assert "authority" in terms
    assert "institution" in terms, terms
    assert "oversee" not in terms, "a verb reached the vocabulary"
    assert "the" not in terms


def test_a_web_address_contributes_no_vocabulary() -> None:
    """Its path segments arrive tagged as nouns."""
    linked = (
        "The supervisory authority oversees the institutions. See "
        "https://www.example.de/publikationen/fokus/im/en for the findings."
    )

    terms = read(chunk(linked))[0].lemmas

    assert "publikationen" not in terms, terms
    assert "fokus" not in terms
    assert "authority" in terms


def test_a_foreign_function_word_is_not_german_vocabulary() -> None:
    """In a language that capitalises every noun, a lower-case one is foreign."""
    mixed = (
        "Die Bundesanstalt beaufsichtigt die Institute. Der Bericht heißt "
        "the annual report of the supervisory authority."
    )

    terms = read(chunk(mixed), language="de")[0].lemmas

    assert "the" not in terms, terms
    assert "of" not in terms
    assert "bundesanstalt" in terms


def test_every_passage_comes_back_read_in_the_order_it_was_given() -> None:
    """The batching is per language, so the order has to be restored."""
    read_back = read(chunk(GERMAN), chunk(ENGLISH), chunk(GERMAN), language="de")

    assert [one.language for one in read_back] == ["de", "en", "de"]
    assert [one.text for one in read_back] == [GERMAN, ENGLISH, GERMAN]
