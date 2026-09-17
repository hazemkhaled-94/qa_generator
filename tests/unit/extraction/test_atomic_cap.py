"""The cap EXTRACTION_MIN_OTHER_SHARE puts on a passage's atomic facts.

A floor on the share of a passage's facts that are not atomic is a cap on
the atomic ones, because a passage yields a fixed number of digests however
much it says. This is the arithmetic of that, and which facts survive it.

Read without a cap one corpus gave 18.4 atomic facts a passage against two
digests - atomic was 91.9% of everything extracted, and an author list
became fifty facts of the form "X wrote the original edition".
"""

from __future__ import annotations

import pytest

from database.qa_generator import FactKind, Rejection
from extraction.config import atomic_cap
from extraction.models import CheckedFact
from extraction.service import over_cap


def fact(
    statement: str,
    *,
    kind: str = FactKind.ATOMIC,
    units: tuple[str, ...] = (),
    validated: bool = True,
) -> CheckedFact:
    """One checked fact, as the cap reads it."""
    return CheckedFact(
        statement=statement,
        evidence_text=statement,
        extraction_method="llm",
        validated=validated,
        rejection_code=None if validated else Rejection.ASSERTS_NOTHING,
        validation_error=None if validated else "nothing",
        kind=kind,
        units_statement=list(units),
    )


# ── What the floor works out to ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        # Two digests beside n atomic facts is a share of 2/(n+2), so the
        # largest n holding the others at `share` is 2(1-share)/share.
        (0.10, 18),
        (0.20, 8),
        (0.25, 6),
        (0.33, 4),
        (0.50, 2),
    ],
)
def test_the_floor_becomes_a_cap_on_the_atomic_facts(share, expected) -> None:
    """The one number that decides how much of a passage becomes facts."""
    assert atomic_cap(2, share) == expected


def test_the_measured_share_of_one_corpus_reproduces_its_own_figure() -> None:
    """18.4 atomic a passage against two digests is a tenth of everything."""
    assert atomic_cap(2, 0.10) == 18


@pytest.mark.parametrize("share", [0.0, 1.0, -1.0, 2.0])
def test_a_share_that_is_not_a_share_caps_nothing(share) -> None:
    """Rather than refusing every fact, or none of them, quietly."""
    assert atomic_cap(2, share) is None


def test_a_deployment_writing_no_digests_caps_nothing() -> None:
    """There is no other kind for a share of the facts to be."""
    assert atomic_cap(0, 0.33) is None


def test_the_cap_is_never_below_one() -> None:
    """A passage read for nothing is a passage not worth the call."""
    assert atomic_cap(2, 0.99) == 1


# ── Which facts it refuses ─────────────────────────────────────────────────


def test_nothing_is_refused_under_the_cap() -> None:
    """A passage that yielded few facts is left alone."""
    facts = [fact("One."), fact("Two.")]

    assert over_cap(facts, 4) == set()


def test_no_cap_refuses_nothing() -> None:
    """None is how EXTRACTION_MIN_OTHER_SHARE turns the whole thing off."""
    assert over_cap([fact(f"{n}.") for n in range(20)], None) == set()


def test_the_facts_asserting_a_value_are_the_ones_kept() -> None:
    """A fact carrying a number is what a checkable question is written from.

    The same rule question generation offers facts by, so the cap keeps what
    the writer would have reached for anyway.
    """
    facts = [
        fact("Boilerplate one."),
        fact("The device weighs 4 kg.", units=("4",)),
        fact("Boilerplate two."),
        fact("It ships in March 2026.", units=("2026",)),
    ]

    # Positions 0 and 2 carry no unit, so they are the two that go.
    assert over_cap(facts, 2) == {0, 2}


def test_ties_go_to_the_order_the_model_wrote_them_in() -> None:
    """So one passage read twice keeps the same facts."""
    facts = [fact(f"Claim {n}.") for n in range(5)]

    assert over_cap(facts, 2) == {2, 3, 4}


def test_only_the_atomic_facts_compete_for_the_budget() -> None:
    """A summary is what the floor exists to make room for."""
    facts = [
        fact("A summary.", kind=FactKind.SUMMARY),
        fact("- An outline", kind=FactKind.OUTLINE),
        fact("One."),
        fact("Two."),
        fact("Three."),
    ]

    # Only positions 2, 3 and 4 are atomic, so a cap of 2 takes one of them
    # and never touches the digests.
    assert over_cap(facts, 2) == {4}


def test_a_fact_a_check_already_refused_is_not_refused_again() -> None:
    """It is not taking a place from anything, and its code says why it went.

    Overwriting it with `over_cap` would lose the real reason.
    """
    facts = [
        fact("Refused already.", validated=False),
        fact("One."),
        fact("Two."),
        fact("Three."),
    ]

    refused = over_cap(facts, 2)

    assert 0 not in refused
    assert refused == {3}
