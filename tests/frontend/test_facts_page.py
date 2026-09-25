"""The Facts page, run against a scripted backend.

What the page does with an answer, including the answers that are refusals.
The rules it shares with every other page are in test_views.py; this covers
what only extraction has: four readings of one corpus, and a check per
reading.
"""

from __future__ import annotations

import pytest
from pages import FACT, FACT_QUALITY, Answers, answers, changed, resting

pytestmark = pytest.mark.frontend


def page_of(*facts: dict) -> dict:
    """One page of facts, and the total behind it."""
    return {"total": len(facts), "facts": list(facts)}


#: The view this module is about. `page` in conftest.py reads it.
VIEW = "facts"


# ── The four readings ─────────────────────────────────────────────────────


def test_every_reading_is_offered_as_a_filter(page) -> None:
    """Atomic, summary, outline and bridge, each choosable on its own."""
    offered = page().options("facts-kind")

    assert offered == [
        "All readings",
        "atomic",
        "summary",
        "outline",
        "bridge",
    ]


def test_each_reading_is_explained_where_it_is_chosen(page) -> None:
    """A reader picking one should not have to know what it means."""
    said = page().tooltip("facts-kind")

    for reading in ("Atomic", "Summary", "Outline", "Bridge"):
        assert reading in said


def test_choosing_a_reading_narrows_what_is_asked_for(open_view) -> None:
    """The backend filters; the page does not."""
    client = Answers(**answers())

    page = open_view("facts", client).choose("facts-kind", "summary")

    assert page.raised == []
    assert [one for one in client.asked if one[2].get("kind") == "summary"]


def test_the_corpus_is_counted_by_reading(page) -> None:
    """Each reading is a row in the fold, whether or not any were drawn."""
    tables = page(
        fact_quality=changed(
            FACT_QUALITY, total=4, kinds={"atomic": 2, "summary": 1, "bridge": 1}
        )
    ).tables()

    for reading in ("Atomic facts", "Summary facts", "Outline facts", "Bridge facts"):
        assert reading in tables


def test_the_kind_a_fact_was_stored_under_is_shown_in_the_table(page) -> None:
    """So a reader can tell a summary from a claim without opening it."""
    tables = page(facts=page_of(changed(FACT, kind="summary"))).tables()

    assert "Summary" in tables


def test_a_reading_the_page_has_no_description_for_is_still_counted(page) -> None:
    """A kind added to the backend should not vanish from the figures."""
    drawn = page(facts=page_of(changed(FACT, kind="invented")))

    assert drawn.raised == []
    assert "invented" in drawn.tables()


# ── The checks ────────────────────────────────────────────────────────────


def test_every_check_is_listed_whether_or_not_anything_failed_it(page) -> None:
    """A check missing from a list cannot be told from one nobody wrote."""
    tables = page().tables()

    for check in (
        "Citation resolves",
        "Not copied",
        "Asserts something",
        "Exactly one claim",
        "Nothing invented",
        "No dangling pronouns",
        "Shorter than its passage",
        "Two points or more",
        "Rests on two passages",
    ):
        assert check in tables, check


def test_each_check_says_which_kinds_face_it(page) -> None:
    """An outline is exempt from the verb check, which the row has to say."""
    tables = page().tables()

    assert "Applies to atomic, summary, bridge" in tables
    assert "Applies to every kind" in tables


def test_a_check_nothing_failed_reads_as_ok(page) -> None:
    """And one something failed does not."""
    tables = page(
        fact_quality=changed(FACT_QUALITY, total=2, validated=1, rejected={"copied": 1})
    ).tables()

    assert "Attention" in tables


def test_a_rejection_code_the_page_has_never_heard_of_is_still_reported(page) -> None:
    """A check added to the backend should not disappear from the report."""
    drawn = page(fact_quality=changed(FACT_QUALITY, total=2, rejected={"brand_new": 1}))

    assert drawn.raised == []
    assert "brand_new" in drawn.tables()


def test_a_refused_fact_names_the_check_it_failed(page) -> None:
    """In the table, so a reader need not open it to know why."""
    tables = page(
        facts=page_of(
            changed(
                FACT,
                validated=False,
                rejection_code="unsupported_addition",
                validation_error="added 70%",
            )
        )
    ).tables()

    assert "Nothing invented" in tables


def test_the_reason_a_fact_was_refused_is_on_the_fact(page) -> None:
    """Opened, it says what it added rather than only that it added."""
    tables = (
        page(
            facts=page_of(
                changed(
                    FACT,
                    validated=False,
                    rejection_code="unsupported_addition",
                    validation_error="added 70%",
                    units_added=["70%"],
                )
            )
        )
        .select(0)
        .tables()
    )

    assert "added 70%" in tables
    assert "70%" in tables


# ── Decomposition, which no single fact can fail ──────────────────────────


def test_decomposition_is_not_measured_over_a_mixed_set(page) -> None:
    """A summary keeps every claim its passage carried, on purpose."""
    tables = page().tables()

    assert "Measured over atomic facts" in tables


def test_decomposition_is_measured_once_one_reading_is_chosen(open_view) -> None:
    """With Atomic picked, the ratio is a number and has a verdict."""
    page = open_view("facts").choose("facts-kind", "atomic")

    assert "2.0×" in page.tables()


# ── What a fact rests on ──────────────────────────────────────────────────


def test_a_fact_resting_on_several_passages_names_them_all(page) -> None:
    """A bridge cites a span in each of the passages it was shown."""
    bridge = changed(
        FACT,
        kind="bridge",
        passages=[FACT["passages"][0], resting(12, 9, 1, page_from=4)],
    )
    tables = page(facts=page_of(bridge)).select(0).tables()

    assert "3, 9" in tables


def test_the_listing_carries_the_passages_so_none_are_asked_for(open_view) -> None:
    """One request for the page, not one per row on it."""
    client = Answers(**answers())

    open_view("facts", client).select(0)

    assert not client.calls("passage")
