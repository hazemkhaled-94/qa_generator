"""The corpus and the fitter settings the topic tests share."""

from __future__ import annotations

import pytest

from topic_modelling.models import PassageVocabulary

#: Two clear subjects, one passage per text.
CORPUS = (
    "lieferung versand transport monatlich paketdienst",
    "lieferung versand transport jaehrlich spedition",
    "wartung reparatur instandhaltung ersatzteil pruefung",
    "wartung reparatur instandhaltung wartungsplan intervall",
)


@pytest.fixture
def settings() -> dict:
    """The fitter settings the topic tests run under."""
    return {
        "num_topics": 2,
        "passes": 5,
        "random_state": 42,
        "top_terms": 5,
        "min_weight": 0.1,
        "no_below": 2,
        "no_above": 0.9,
    }


@pytest.fixture
def passages() -> list[PassageVocabulary]:
    """The corpus as the fitter takes it."""
    return [
        PassageVocabulary(id=index, lemmas=text.split())
        for index, text in enumerate(CORPUS, start=1)
    ]
