"""The corpus, the fitter settings and the drivers the topic tests share."""

from __future__ import annotations

import pytest
from drivers import FitterDriver, LabellerDriver, ServiceDriver, corpus

from topic_modelling.models import PassageVocabulary

#: Two clear subjects, one passage per text.
CORPUS = (
    "lieferung versand transport monatlich paketdienst",
    "lieferung versand transport jaehrlich spedition",
    "wartung reparatur instandhaltung ersatzteil pruefung",
    "wartung reparatur instandhaltung wartungsplan intervall",
)

#: The same two subjects in English, for the tests that need two languages.
ENGLISH = (
    "delivery dispatch transport monthly courier",
    "delivery dispatch transport yearly haulier",
    "maintenance repair upkeep spare inspection",
    "maintenance repair upkeep schedule interval",
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
    return corpus(*CORPUS)


@pytest.fixture
def english() -> list[PassageVocabulary]:
    """The English corpus, numbered clear of the German one."""
    return corpus(*ENGLISH, first_id=101)


@pytest.fixture
def fitter(settings, passages) -> FitterDriver:
    """The fitter over the shared corpus."""
    return FitterDriver(settings, passages)


@pytest.fixture
def fitting(fitter):
    """One fitted model over the shared corpus."""
    return fitter.fit("de")


@pytest.fixture
def service(settings, passages):
    """Builds the flow over whichever corpora a test asks for."""

    def build(
        corpora: dict | None = None,
        texts: dict[int, str] | None = None,
        labeller: LabellerDriver | None = None,
        **tuning: object,
    ) -> ServiceDriver:
        """Wires the service over the doubles, `tuning` overriding a setting."""
        return ServiceDriver(
            settings=settings | tuning,
            corpora={"de": passages} if corpora is None else corpora,
            texts=texts,
            labeller=labeller,
        )

    return build
