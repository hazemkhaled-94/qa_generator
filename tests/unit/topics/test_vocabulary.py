"""What reaches the vocabulary a topic model is fitted over.

Document frequency cannot separate subjects from grammar: keeping only
content parts of speech is what separates them, and it also folds the
inflections German spreads one term across.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.nlp


def lemmas(text: str, language: str) -> list[str]:
    """Reads one text's stored vocabulary."""
    from nlp.analysis import read

    return next(iter(read([text], language)))[1]


def test_german_grammar_is_dropped_and_its_subjects_kept() -> None:
    """Articles, auxiliaries and prepositions are not vocabulary."""
    german = lemmas("Die Lieferung wird vor dem Versand wie folgt geprueft", "de")
    assert not {"die", "wird", "vor", "dem", "wie"} & set(german), german
    assert "lieferung" in german, german


def test_english_grammar_is_dropped_and_its_subjects_kept() -> None:
    """The same, in the English pipeline."""
    english = lemmas("The delivery is checked before the dispatch.", "en")
    assert not {"the", "is", "before"} & set(english), english
    assert "delivery" in english, english


def test_inflections_fold_to_one_term() -> None:
    """The frequency filter can only work on folded forms."""
    folded = lemmas("Institut Instituts Institute", "de")
    assert folded.count("institut") >= 2, folded


def test_a_lemma_the_pipeline_could_not_read_is_not_stored() -> None:
    """The lemmatiser's placeholder for a foreign token is not a term."""
    mixed = lemmas("Die Bafin prueft the risk on the market genau.", "de")
    assert "--" not in mixed, mixed
    assert all(term.isalpha() for term in mixed), mixed


def test_embedded_foreign_grammar_is_dropped_and_its_content_kept() -> None:
    """German capitalises every noun, so a lower-case one is another language."""
    embedded = lemmas(
        "Quelle: Baker and Davis, Economic Policy Index. Der Bericht "
        "'Risks in the Focus of Bafin' nennt die Kosten mit dem Markt.",
        "de",
    )
    assert not {"the", "and", "of", "with"} & set(embedded), embedded
    assert {"economic", "risks", "focus"} <= set(embedded), embedded


def test_a_german_sentence_keeps_every_noun_it_has() -> None:
    """The rule above takes no German noun with it."""
    german_only = lemmas("Die Lieferung erreicht den Hafen puenktlich.", "de")
    assert {"lieferung", "hafen"} <= set(german_only), german_only


def test_a_table_is_vocabulary_too() -> None:
    """Its headings and cell values are what it is about."""
    grid = lemmas(
        "Tabelle 1: Planstellenübersicht\n\n| Besoldungsgruppe | Anzahl |\n"
        "| --- | --- |\n| Oberregierungsrat | 12 |\n",
        "de",
    )
    assert {"besoldungsgruppe", "anzahl"} <= set(grid), grid
    assert all(term.isalpha() for term in grid), grid


def test_a_web_address_is_not_vocabulary() -> None:
    """Markdown brackets stop the tokenizer seeing a link as one token."""
    linked = lemmas(
        "[Risks in Bafin's Focus](https://www.bafin.de/EN/die-bafin/"
        "publikationen-daten/risiken-im-fokus/Fokusrisiken.html)",
        "en",
    )
    assert not {"publikationen", "fokus", "im", "en", "die"} & set(linked), linked
    assert "risk" in linked, "the link text is still vocabulary"


def test_no_language_is_judged_on_case_when_none_is_named(monkeypatch) -> None:
    """NLP_CAPITALISED_NOUNS names the languages that capitalise every noun.

    Empty is valid: a deployment reading only languages that capitalise
    nothing names none, and the rule stops applying rather than misfiring.
    """
    from nlp import analysis

    monkeypatch.setenv("NLP_CAPITALISED_NOUNS", "")
    analysis._capitalises_nouns.cache_clear()
    try:
        kept = lemmas("Der Bericht 'Risks in the Focus' nennt die Kosten.", "de")
        assert {"the"} & set(kept), kept
    finally:
        analysis._capitalises_nouns.cache_clear()
