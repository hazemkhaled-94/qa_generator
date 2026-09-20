"""Which words point at something outside the text they sit in.

Three readings, because no one of them covers both languages: `PronType=Dem`
off the morphology, a noun made definite AND counted, and the anaphoric
adjectives that carry no feature at all. Both configured languages are
checked, because a rule that held only for the one the corpus happened to be
in is the thing this project is not allowed to be.
"""

from __future__ import annotations

import pytest

from nlp.analysis import interrogatives, pointing

pytestmark = pytest.mark.nlp


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Wie groß ist der Unterschied zwischen diesen beiden Lehrplänen?", "de"),
        ("Was wurde in dieser Version geändert?", "de"),
        ("How long is the gap between these two editions?", "en"),
        ("What changed in this version?", "en"),
    ],
)
def test_a_pointing_word_is_found(text: str, language: str) -> None:
    """The determiner carries it, and the noun beside it hides it."""
    assert pointing(text, language), "nothing was read as pointing"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Wie groß ist der Unterschied zwischen den beiden Lehrplänen?", "de"),
        ("How large is the gap between the two editions?", "en"),
    ],
)
def test_a_definite_count_points_without_a_demonstrative(
    text: str, language: str
) -> None:
    """The reading that made this gate fire zero times over 3,131 questions.

    `den beiden` is an article plus `PronType=Ind` and `the two` is an
    article plus `NumType=Card`, so neither is `PronType=Dem` and the
    measurement never saw either. A set of a known size the question never
    introduced points outward as surely as `diesen beiden` does.
    """
    assert pointing(text, language), "a definite count was read as pointing at nothing"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("How do the aforementioned parties differ?", "en"),
        ("What does the latter require?", "en"),
    ],
)
def test_an_anaphoric_adjective_points(text: str, language: str) -> None:
    """These carry no feature at all: every tagset reads them as adjectives."""
    assert pointing(text, language), "an anaphoric adjective was not read"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Wie lange dauert die Beantwortung einer Standardanfrage?", "de"),
        ("How long is allowed for answering a standard support request?", "en"),
    ],
)
def test_a_question_pointing_nowhere_is_clean(text: str, language: str) -> None:
    """The measurement must not fire on an ordinary question."""
    assert pointing(text, language) == ()


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Wie lange dauert die Beantwortung einer Standardanfrage?", "de"),
        ("How long is allowed for answering a standard support request?", "en"),
        ("Welche Verfahren gelten für Abnahmekriterien?", "de"),
        ("Which techniques apply to acceptance criteria?", "en"),
    ],
)
def test_a_question_word_is_found_under_either_naming(text: str, language: str) -> None:
    """Penn and STTS say so in the tag; a UD tagset says so in the morphology.

    English `What` carries no PronType at all and German `Wie` carries no W
    tag, so reading only one of the two conventions reports nothing for one
    of the two languages that ship.
    """
    assert interrogatives(text, language), "no question word was read"
