"""What a question's own wording says, read without a model.

Two of the three phrasing judgements moved off the verifier here, and the
measurement that sent them is in `evaluation/README.md`: answered inside the
call that reads the passages they were right 7 times out of 7 in English and
6 out of 12 in German.

Every case below is one of the nineteen labelled ones, so what these assert
and what `make test-eval` scores are the same cases. The eval module prints;
this one gates, because a rule is not a matter of taste.
"""

from __future__ import annotations

import pytest

from question_generation.gates import cites_source, subject

pytestmark = pytest.mark.nlp


@pytest.mark.parametrize(
    ("question", "language"),
    [
        (
            (
                "Wie unterscheiden sich in Abschnitt 2.2 die Themen zur "
                "Testschätzung und zur Fehlerbehebung?"
            ),
            "de",
        ),
        ("Welches Thema behandelte Kapitel 5 in der älteren Fassung?", "de"),
        ("In welchem Monat wurde die Referenz [R22] aufgerufen?", "de"),
        ("Wie definiert Beck 2003 Refactoring in der Entwicklung?", "de"),
        (
            (
                "Unter welchen Einteilungen werden Testverfahren in diesem "
                "Lehrplan klassifiziert?"
            ),
            "de",
        ),
        ("What does section 4 say about weekend cover?", "en"),
    ],
)
def test_a_named_source_is_settled_without_a_model(
    question: str, language: str
) -> None:
    """A numbered division, a bracketed reference, an author with a year.

    Four of these are cases gpt-4.1 got wrong inside the reading call, and
    none of them needs a reading: they are patterns.
    """
    assert cites_source(question, language) is True


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # The standard is what the question is ABOUT, not where the answer
        # is - and `die Norm ISO/IEC 20246` in the same shape IS a source.
        # Nothing structural separates them, so the rules must not guess.
        ("Warum wird ISO/IEC/IEEE 29119-4 in diesem Zusammenhang erwähnt?", "de"),
        ("Wer trägt die Verantwortung für den vierteljährlichen Risikobericht?", "de"),
        ("How many faults were reported to the site manager in 2025?", "en"),
        ("Within how many hours is an urgent support request answered?", "en"),
    ],
)
def test_what_needs_a_reading_is_left_to_one(question: str, language: str) -> None:
    """None, never False: absence of a pattern is not evidence of absence."""
    assert cites_source(question, language) is None


def test_a_compound_naming_a_document_is_not_a_citation() -> None:
    """`Risikobericht` is a thing the corpus is about; `Bericht` is one it is.

    Read by lemma and never as a substring, because German compounds put one
    inside the other.
    """
    assert (
        cites_source(
            "Wer trägt die Verantwortung für den vierteljährlichen Risikobericht?",
            "de",
        )
        is None
    )


def test_attributing_to_something_unnamed_is_not_naming_a_source() -> None:
    """`laut diesen Angaben` points at something; it cites nothing.

    That question fails a different gate - it does not stand on its own -
    and running the two together would file it under the wrong one.
    """
    assert (
        cites_source(
            "Wie unterscheiden sich der Lehrplaninhalt und ein "
            "Testteammitglied laut diesen Angaben?",
            "de",
        )
        is None
    )


@pytest.mark.parametrize(
    ("question", "language", "holds"),
    [
        (
            "Wie viele Testfälle verlangt die Grenzwertanalyse bei drei Partitionen?",
            "de",
            "Grenzwertanalyse",
        ),
        ("How many faults were reported in 2025?", "en", "faults"),
        ("Welche Reviewverfahren beschreibt die Norm ISO/IEC 20246?", "de", "20246"),
    ],
)
def test_the_subject_is_copied_off_the_parse(
    question: str, language: str, holds: str
) -> None:
    """Containment, not equality: two correct readings disagree about edges."""
    assert holds.casefold() in subject(question, language).casefold()


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("What specific components are included?", "en"),
        ("Für welche Kriterien gelten die Anforderungen?", "de"),
    ],
)
def test_a_question_naming_nothing_has_no_subject(question: str, language: str) -> None:
    """Every noun a bare word, with nothing saying whose or which."""
    assert subject(question, language) == ""
