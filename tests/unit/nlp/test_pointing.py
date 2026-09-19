"""Which words point at something outside the text they sit in.

Read off `PronType=Dem`, which every Universal Dependencies tagset marks, so
a language added to NLP_MODELS brings its own without a word list here. Both
configured languages are checked, because a rule that held only for the one
the corpus happened to be in is the thing this project is not allowed to be.
"""

from __future__ import annotations

import pytest

from nlp.analysis import demonstratives, interrogatives

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
    assert demonstratives(text, language), "nothing was read as pointing"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Wie lange dauert die Beantwortung einer Standardanfrage?", "de"),
        ("How long is allowed for answering a standard support request?", "en"),
    ],
)
def test_a_question_pointing_nowhere_is_clean(text: str, language: str) -> None:
    """The measurement must not fire on an ordinary question."""
    assert demonstratives(text, language) == ()


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
