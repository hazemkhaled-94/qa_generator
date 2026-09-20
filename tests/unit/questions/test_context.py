"""What the writer is told about where each fact came from.

A sample may hold passages from two documents, and the writer decides one
question over all of them. Whether it can tell the two apart is the
difference between a question about a difference and a question with two
answers: two syllabi giving different chapter counts were asked as one
question and answered "once with three, once with eight", which is a row
nobody can be marked against.
"""

from __future__ import annotations

from factories import group, source


def test_one_document_is_shown_the_way_it_always_was() -> None:
    """Most samples are one document.

    A label carrying no information is one a model reaches for anyway, and a
    prompt that moves for no reason cannot be compared against the last run.
    """
    facts = group(source(1, document="a", passage_id=1, section_path="Support"))

    assert [heading for heading, _ in facts.context] == ["Under: Support\n"]


def test_two_documents_are_told_apart() -> None:
    """The defect this exists for."""
    facts = group(
        source(1, document="a", passage_id=1, section_path="Chapters"),
        source(2, document="b", passage_id=2, section_path="Chapters"),
    )

    assert [heading for heading, _ in facts.context] == [
        "Document A, under: Chapters\n",
        "Document B, under: Chapters\n",
    ]


def test_a_letter_belongs_to_a_document_not_to_a_passage() -> None:
    """Two passages of one document are one source, not two."""
    facts = group(
        source(1, document="a", passage_id=1),
        source(2, document="a", passage_id=2),
        source(3, document="b", passage_id=3),
    )

    assert [heading.split(",")[0].strip() for heading, _ in facts.context] == [
        "Document A",
        "Document A",
        "Document B",
    ]


def test_a_document_with_no_heading_is_still_named() -> None:
    """A passage the parser found no heading for is still one of the sources.

    The label is what says two facts come from two documents, and a missing
    heading does not take it away.
    """
    facts = group(
        source(1, document="a", passage_id=1, section_path=None),
        source(2, document="b", passage_id=2, section_path=None),
    )

    assert [heading for heading, _ in facts.context] == [
        "Document A\n",
        "Document B\n",
    ]


def test_the_passage_text_is_carried_beside_its_heading() -> None:
    """The heading is fenced off rather than run into the text.

    A model shown the two as one block asks about the heading.
    """
    facts = group(source(1, document="a", passage_id=1, section_path="Support"))

    heading, text = facts.context[0]
    assert heading.endswith("\n")
    assert text == facts.facts[0].passage_text
