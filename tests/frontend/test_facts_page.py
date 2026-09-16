"""The Facts page, run against a scripted backend.

A page that raises renders a Streamlit traceback and nothing else, so every
test here asserts the page rendered as well as what it said.
"""

from __future__ import annotations

import pytest
import requests
from conftest import Answers
from facts_page import FACT, FactsPage, fact, page_of, quality, resting

pytestmark = pytest.mark.frontend

#: What the backend answers when it is down.
UNREACHABLE = requests.exceptions.ConnectionError("connection refused")


@pytest.fixture
def page(run_view):
    """Runs the Facts view against whatever the backend is scripted to say."""

    def run(**changed) -> FactsPage:
        """Runs the page with these answers replacing the defaults."""
        return FactsPage(
            run_view("facts", catalog_api=Answers(**FactsPage.answers(**changed)))
        )

    return run


def test_an_empty_corpus_says_what_to_do_first(page) -> None:
    """The first thing a new deployment shows."""
    drawn = page(facts=page_of())

    assert not drawn.raised, drawn.raised
    assert "No facts yet" in drawn.text(), drawn.text()


def test_a_backend_that_is_down_is_reported_rather_than_raised(page) -> None:
    """The reader is told, not shown a stack."""
    drawn = page(facts=UNREACHABLE, fact_quality=UNREACHABLE)

    assert not drawn.raised, drawn.raised
    assert drawn.text()


def test_every_reading_is_offered_as_a_filter(page) -> None:
    """A reader looks at one kind at a time, or at all of them."""
    drawn = page()

    assert drawn.readings() == [
        "All kinds",
        "Atomic",
        "Summary",
        "Outline",
        "Bridge",
    ]


def test_each_reading_is_explained_where_it_is_chosen(page) -> None:
    """A vocabulary a reader cannot look up is a vocabulary nobody uses."""
    drawn = page()

    for said in ("one sentence", "whole passage", "one bullet each", "no single"):
        assert said in drawn.helps(), said


def test_the_corpus_is_counted_by_reading(page) -> None:
    """How much of the corpus each reading covers."""
    drawn = page(
        facts=page_of(fact(), fact(id=2, kind="summary")),
        fact_quality=quality(total=2, validated=2, kinds={"atomic": 1, "summary": 1}),
    )

    figures = drawn.metrics()
    assert figures["Atomic"] == "1"
    assert figures["Summary"] == "1"
    assert figures["Outline"] == "0"
    assert figures["Bridge"] == "0"


def test_the_kind_a_fact_was_stored_under_is_shown_in_the_table(page) -> None:
    """So a reader can tell a summary from a claim without opening it."""
    drawn = page(
        facts=page_of(fact(), fact(id=2, kind="bridge", statement="A bridge.")),
        fact_quality=quality(total=2, kinds={"atomic": 1, "bridge": 1}),
    )

    assert "Atomic" in drawn.tables()
    assert "Bridge" in drawn.tables()


def test_choosing_a_reading_narrows_what_is_asked_for(run_view) -> None:
    """Every figure and row below is the one kind, not all of them."""
    client = Answers(**FactsPage.answers())
    drawn = FactsPage(run_view("facts", catalog_api=client)).choose_reading("Summary")

    assert not drawn.raised, drawn.raised
    listed = [call for call in client.asked if call[0] == "facts"]
    assert listed[-1][2]["kind"] == "summary", listed[-1]


def test_every_check_is_listed_whether_or_not_anything_failed_it(page) -> None:
    """A check missing from a list cannot be told apart from one nobody wrote."""
    drawn = page()

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
        assert check in drawn.tables(), check


def test_each_check_says_which_kinds_face_it(page) -> None:
    """A check no kind in view faces reads 0 because nothing asked it."""
    drawn = page()

    assert "summary, outline" in drawn.tables()
    assert "atomic, bridge" in drawn.tables()


def test_decomposition_is_not_measured_over_a_mixed_set(page) -> None:
    """A summary keeps the claims its passage carried on purpose."""
    drawn = page()

    assert "Only atomic facts are decomposed" in drawn.tables()


def test_decomposition_is_measured_once_one_reading_is_chosen(run_view) -> None:
    """Which is the figure that says whether the model decomposed at all."""
    drawn = FactsPage(
        run_view("facts", catalog_api=Answers(**FactsPage.answers()))
    ).choose_reading("Atomic")

    assert "2.0×" in drawn.tables(), drawn.tables()


def test_a_reading_the_page_has_no_description_for_is_still_counted(page) -> None:
    """Added to the backend without being added here."""
    drawn = page(fact_quality=quality(rejected={"invented_code": 1}))

    assert "invented_code" in drawn.tables()
    assert "no description" in drawn.tables()


def test_a_refused_fact_names_the_check_it_failed(page) -> None:
    """Kept rather than discarded, so a reader can see why."""
    refused = fact(
        id=2,
        kind="outline",
        statement="- One point",
        validated=False,
        rejection_code="not_listed",
        validation_error="the outline holds 1 point(s)",
    )
    drawn = page(
        facts=page_of(refused),
        fact_quality=quality(
            total=1, validated=0, kinds={"outline": 1}, rejected={"not_listed": 1}
        ),
    )

    assert "Two points or more" in drawn.tables()


def test_a_fact_resting_on_several_passages_names_them_all(page) -> None:
    """One passage's evidence would not say what the claim was drawn from."""
    drawn = page(
        facts=page_of(
            fact(
                id=7,
                kind="bridge",
                statement="A bridge.",
                passages=[FACT["passages"][0], resting(22, ordinal=9, position=1)],
            )
        ),
        fact_quality=quality(kinds={"bridge": 1}),
    )

    assert "Rests on passages: 3 (id 11), 9 (id 22)" in drawn.text(), drawn.text()


def test_the_listing_carries_the_passages_so_none_are_asked_for(run_view) -> None:
    """Every kind names its passages on the row; no second call is made."""
    client = Answers(**FactsPage.answers())
    run_view("facts", catalog_api=client)

    assert not [call for call in client.asked if call[0] == "fact_passages"]
