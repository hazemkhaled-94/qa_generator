"""How many claims a sentence reads as.

A German obligation is stated with a modal and a participle. de_core_news_sm
tags no finite verb in that shape at all; a failure here means NLP_MODELS has
been pointed at a smaller model.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.nlp


@pytest.mark.parametrize(
    "sentence",
    [
        "Ein Risikobericht muss mindestens vierteljährlich erstellt werden.",
        "Bei abweichenden Voten muss der Kredit abgelehnt werden.",
        "Die Geschäftsleitung muss für die Umsetzung Sorge tragen.",
        "Die vereinfachte Umsetzung muss unter Risikogesichtspunkten vertretbar sein.",
        "Das interne Kontrollsystem umfasst eine Risikocontrolling-Funktion.",
    ],
)
def test_a_german_obligation_is_one_claim(sentence: str) -> None:
    """A modal and its participle read as a single claim."""
    from nlp.analysis import claim

    found = claim(sentence, "de").predicates
    assert found == 1, f"{found} claim(s) read in {sentence!r}; expected 1"


def test_two_coordinated_clauses_are_two_claims() -> None:
    """A sentence joining two finite verbs reads as two."""
    from nlp.analysis import claim

    sentence = "Der Antrag wird geprüft und das Ergebnis wird mitgeteilt."
    assert claim(sentence, "de").predicates == 2


def test_a_noun_phrase_asserts_nothing() -> None:
    """A phrase with no finite verb reads as no claim."""
    from nlp.analysis import claim

    phrase = "Die Erhöhung der Eigenkapitalquote auf 12,5 Prozent"
    assert claim(phrase, "de").predicates == 0
