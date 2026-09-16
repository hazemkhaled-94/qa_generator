"""The verdicts the checks have always reached, pinned as a table.

One row per candidate somebody has actually seen, with the verdict it is
meant to get. The unit tests say why a check exists; this says what the whole
set of them does to a corpus, so a change to one check that quietly moves
another shows up as a diff rather than as a number on a page.

Every row here is a case that mattered: a parser limitation, a false refusal
that was fixed, or a failure mode the prompt exists to prevent.
"""

from __future__ import annotations

import pytest
from drivers import Checker, group, passage

from database.qa_generator import FactKind, Rejection

pytestmark = pytest.mark.nlp

#: The two-sentence English passage, whose first sentence carries two claims.
PROSE = (
    "The device weighs 4 kg and runs for 12 hours. "
    "It arrives in March 2026 from the Hamburg plant."
)

#: A German passage, which is the other language this deployment reads.
GERMAN = (
    "Im Jahr 2026 überwacht die Bafin die Kreditrisiken der Institute. "
    "Ein Risikobericht muss mindestens vierteljährlich erstellt werden."
)

#: (name, statement, cited sentence, expected code or None), over PROSE.
ATOMIC: tuple[tuple[str, str, int, str | None], ...] = (
    ("one claim, its own words", "The device weighs 4 kg.", 0, None),
    ("the other claim of the same sentence", "The device runs for 12 hours.", 0, None),
    (
        "a quoted sentence carrying one claim",
        "It arrives in March 2026 from the Hamburg plant.",
        1,
        Rejection.UNRESOLVED_REFERENCE,
    ),
    (
        "the sentence copied whole",
        "The device weighs 4 kg and runs for 12 hours.",
        0,
        Rejection.COPIED,
    ),
    (
        "two claims joined by and",
        "The device weighs 4 kg and it runs for 12 hours.",
        0,
        Rejection.NOT_ATOMIC,
    ),
    ("a noun phrase", "The 4 kg device.", 0, Rejection.ASSERTS_NOTHING),
    ("a heading", "Delivery and dispatch", 0, Rejection.ASSERTS_NOTHING),
    (
        "a number the source never gave",
        "The device weighs 7 kg.",
        0,
        Rejection.UNSUPPORTED_ADDITION,
    ),
    (
        "a name the source never gave",
        "The Bundesbank delivers the device in March 2026.",
        1,
        Rejection.UNSUPPORTED_ADDITION,
    ),
    (
        "a dangling pronoun",
        "It arrives in March 2026.",
        1,
        Rejection.UNRESOLVED_REFERENCE,
    ),
    (
        "a subject taken from the other sentence",
        "The device arrives in March 2026.",
        1,
        None,
    ),
)

#: The same table in German, where the parser behaves differently enough to
#: be worth its own rows.
GERMAN_ATOMIC: tuple[tuple[str, str, int, str | None], ...] = (
    (
        "a modal passive, which the small pipeline reads as no claim at all",
        "Ein Risikobericht muss vierteljährlich erstellt werden.",
        1,
        None,
    ),
    (
        "a year with a full stop attached is the same year",
        "Die Bafin überwacht die Kreditrisiken im Jahr 2026.",
        0,
        None,
    ),
    (
        "a reflexive belongs to its own verb",
        "Die Überwachung ergibt sich aus den Kreditrisiken.",
        0,
        None,
    ),
    (
        "an invented figure",
        "Ein Risikobericht muss monatlich in 12 Ausfertigungen erstellt werden.",
        1,
        Rejection.UNSUPPORTED_ADDITION,
    ),
)

#: (name, statement, expected code or None), over PROSE as a summary.
SUMMARIES: tuple[tuple[str, str, str | None], ...] = (
    ("a condensed reading", "The device weighs 4 kg. It arrives in March 2026.", None),
    (
        "several claims, which is what a summary is for",
        "The device weighs 4 kg and runs for 12 hours.",
        None,
    ),
    ("the passage copied whole", PROSE, Rejection.NOT_CONDENSED),
    (
        "a label rather than a reading",
        "The 4 kg device from Hamburg.",
        Rejection.ASSERTS_NOTHING,
    ),
    (
        "a figure the passage does not carry",
        "The device weighs 9 kg.",
        Rejection.UNSUPPORTED_ADDITION,
    ),
)

#: (name, points, expected code or None), over PROSE as an outline.
OUTLINES: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    (
        "telegraphic points, which carry no finite verb",
        ("Weighs 4 kg", "Runs for 12 hours"),
        None,
    ),
    (
        "points written as sentences",
        ("It weighs 4 kg.", "It runs for 12 hours."),
        None,
    ),
    ("one point, which is a label", ("Weighs 4 kg",), Rejection.NOT_LISTED),
    ("no points at all", (), Rejection.NOT_LISTED),
    (
        "a point naming a figure the passage does not",
        ("Weighs 9 kg", "Runs for 12 hours"),
        Rejection.UNSUPPORTED_ADDITION,
    ),
)

#: (name, statement, positions, expected code or None), over two passages.
BRIDGES: tuple[tuple[str, str, tuple[int, ...], str | None], ...] = (
    (
        "one claim needing both",
        (
            "Support response times are stated separately for standard and "
            "urgent requests."
        ),
        (0, 1),
        None,
    ),
    (
        "a value carried by the second passage",
        "An urgent request waits 4 hours.",
        (0, 1),
        None,
    ),
    (
        "a claim one passage states alone",
        "Standard requests are answered within 48 hours.",
        (0,),
        Rejection.NOT_BRIDGING,
    ),
    (
        "arithmetic nobody wrote down",
        "The response times differ by 44 hours.",
        (0, 1),
        Rejection.UNSUPPORTED_ADDITION,
    ),
    (
        "two claims in one bridge",
        "Standard requests wait 48 hours and urgent requests wait 4 hours.",
        (0, 1),
        Rejection.NOT_ATOMIC,
    ),
    (
        "a claim resting on nothing",
        "Anything at all happens.",
        (),
        Rejection.EVIDENCE_ABSENT,
    ),
)


def verdict(checked) -> str | None:
    """The code a fact was refused under, or None when it passed."""
    return checked.rejection_code


@pytest.fixture(scope="module")
def english() -> Checker:
    """The English passage every atomic and digest row is read against."""
    return Checker(passage(PROSE))


@pytest.fixture(scope="module")
def german() -> Checker:
    """The German passage the German rows are read against."""
    return Checker(passage(GERMAN, language="de"))


@pytest.fixture(scope="module")
def offered():
    """Two passages of two documents, as a topic group offers them."""
    return group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )


@pytest.mark.parametrize(
    ("statement", "cited", "expected"),
    [row[1:] for row in ATOMIC],
    ids=[row[0] for row in ATOMIC],
)
def test_an_atomic_candidate_reaches_its_recorded_verdict(
    english, statement, cited, expected
) -> None:
    """One claim, from one sentence, with nothing added and nothing dangling."""
    assert verdict(english.atomic(statement, cited)) == expected


@pytest.mark.parametrize(
    ("statement", "cited", "expected"),
    [row[1:] for row in GERMAN_ATOMIC],
    ids=[row[0] for row in GERMAN_ATOMIC],
)
def test_a_german_candidate_reaches_its_recorded_verdict(
    german, statement, cited, expected
) -> None:
    """The other language, where the parser behaves differently enough."""
    assert verdict(german.atomic(statement, cited)) == expected


@pytest.mark.parametrize(
    ("statement", "expected"),
    [row[1:] for row in SUMMARIES],
    ids=[row[0] for row in SUMMARIES],
)
def test_a_summary_reaches_its_recorded_verdict(english, statement, expected) -> None:
    """Shorter than its passage, asserting something, inventing nothing."""
    assert verdict(english.summary(statement)) == expected


@pytest.mark.parametrize(
    ("points", "expected"),
    [row[1:] for row in OUTLINES],
    ids=[row[0] for row in OUTLINES],
)
def test_an_outline_reaches_its_recorded_verdict(english, points, expected) -> None:
    """Two points or more, inventing nothing, judged on shape not grammar."""
    assert verdict(english.outline(*points)) == expected


@pytest.mark.parametrize(
    ("statement", "positions", "expected"),
    [row[1:] for row in BRIDGES],
    ids=[row[0] for row in BRIDGES],
)
def test_a_bridge_reaches_its_recorded_verdict(
    english, offered, statement, positions, expected
) -> None:
    """One claim, over two passages, computed from neither."""
    assert verdict(english.bridge(statement, offered, positions)) == expected


def test_every_kind_appears_in_this_table() -> None:
    """A kind with no rows here is a kind nothing pins."""
    covered = {FactKind.ATOMIC, FactKind.SUMMARY, FactKind.OUTLINE, FactKind.BRIDGE}
    assert covered == set(FactKind)


def test_every_refusal_appears_in_this_table() -> None:
    """A code with no row here is a code that could stop firing unnoticed."""
    pinned = {
        expected
        for rows in (ATOMIC, GERMAN_ATOMIC)
        for *_, expected in rows
        if expected
    }
    pinned |= {expected for *_, expected in SUMMARIES if expected}
    pinned |= {expected for *_, expected in OUTLINES if expected}
    pinned |= {expected for *_, expected in BRIDGES if expected}

    assert pinned == set(Rejection), sorted(set(Rejection) - pinned)


def test_an_accepted_fact_carries_a_span_that_resolves_in_its_passage(
    english, offered
) -> None:
    """The invariant every reader of fact_passages relies on."""
    under = english.passage
    for statement, cited, expected in [row[1:] for row in ATOMIC]:
        checked = english.atomic(statement, cited)
        if expected == Rejection.EVIDENCE_ABSENT:
            continue
        ((one,),) = [checked.citations]
        assert one.passage_id == under.id, statement
        assert under.text[one.start : one.end] == checked.evidence_text, statement

    bridged = english.bridge("An urgent request waits 4 hours.", offered)
    by_id = {one.id: one for one in offered}
    assert bridged.evidence_text == "\n".join(
        by_id[one.passage_id].text[one.start : one.end] for one in bridged.citations
    )
