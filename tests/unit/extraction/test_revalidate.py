"""Judging every stored fact again without calling the model.

This is what applies a change to the checks to facts extracted before it.
What the model wrote is the record of one extraction and is kept; what the
checks read off it is derived, and is replaced.
"""

from __future__ import annotations

import pytest
from drivers import Catalogue, group, passage

from database.qa_generator import FactKind, Rejection
from extraction.models import CandidateFact, Cited
from extraction.service import _REJUDGE_BATCH, revalidate

pytestmark = pytest.mark.nlp


def stored(fact_id: int, statement: str, cited=(0,), kind=FactKind.ATOMIC):
    """One stored fact, as the catalogue hands it back."""
    return (
        fact_id,
        [passage()],
        CandidateFact(statement, cited, kind=kind),
        "llm",
    )


def test_nothing_stored_is_nothing_to_judge() -> None:
    """An empty corpus writes no batch at all."""
    catalog = Catalogue()

    assert revalidate(catalog) == 0  # type: ignore[arg-type]
    assert catalog.verdicts == []


def test_every_stored_fact_comes_back_judged() -> None:
    """One verdict per fact, keyed by the id it was read under."""
    catalog = Catalogue().holding(
        stored(1, "The device weighs 4 kg."),
        stored(2, "The device weighs 7 kg."),
    )

    assert revalidate(catalog) == 2  # type: ignore[arg-type]
    assert [fact_id for fact_id, _ in catalog.verdicts] == [1, 2]


def test_the_verdict_is_what_today_checks_read() -> None:
    """A fact that passes and one that does not, both re-read."""
    catalog = Catalogue().holding(
        stored(1, "The device weighs 4 kg."),
        stored(2, "The device weighs 7 kg."),
    )
    revalidate(catalog)  # type: ignore[arg-type]

    passed, refused = (checked for _, checked in catalog.verdicts)
    assert passed.validated, passed.validation_error
    assert refused.rejection_code == Rejection.UNSUPPORTED_ADDITION


def test_a_bridge_is_judged_against_its_whole_group() -> None:
    """Its passages come back from the link table, not from its anchor."""
    offered = group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )
    catalog = Catalogue().holding(
        (
            7,
            offered,
            CandidateFact(
                "Support response times are stated separately for both kinds.",
                (),
                kind=FactKind.BRIDGE,
                passages=(
                    Cited(position=0, sentences=(0,)),
                    Cited(position=1, sentences=(0,)),
                ),
            ),
            "llm",
        )
    )
    revalidate(catalog)  # type: ignore[arg-type]

    ((_, checked),) = catalog.verdicts
    assert checked.kind == FactKind.BRIDGE
    assert [one.passage_id for one in checked.citations] == [one.id for one in offered]
    assert checked.validated, checked.validation_error


def test_a_bridge_left_with_one_passage_is_refused_on_the_second_reading() -> None:
    """Re-chunking took the other side away."""
    offered = group("Standard requests are answered within 48 hours.")
    catalog = Catalogue().holding(
        (
            7,
            offered,
            CandidateFact(
                "Support response times are stated separately for both kinds.",
                (),
                kind=FactKind.BRIDGE,
                passages=(Cited(position=0, sentences=(0,)),),
            ),
            "llm",
        )
    )
    revalidate(catalog)  # type: ignore[arg-type]

    ((_, checked),) = catalog.verdicts
    assert checked.rejection_code == Rejection.NOT_BRIDGING


def test_the_verdicts_are_written_in_batches() -> None:
    """A corpus of facts is not held whole in memory to write it back."""
    catalog = Catalogue().holding(
        *(stored(at, "The device weighs 4 kg.") for at in range(_REJUDGE_BATCH + 3))
    )

    assert revalidate(catalog) == _REJUDGE_BATCH + 3  # type: ignore[arg-type]
    assert [len(batch) for batch in catalog.rejudged] == [_REJUDGE_BATCH, 3]


def test_a_narrowing_reaches_the_read() -> None:
    """`--only` re-judges one document rather than the corpus."""
    catalog = Catalogue()
    revalidate(catalog, "a-condition")  # type: ignore[arg-type]

    assert catalog.narrowed == ["a-condition"]
