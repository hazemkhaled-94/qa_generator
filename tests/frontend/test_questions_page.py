"""The Questions page, run against a scripted backend.

What only this page has: a confidence built out of several readings on
different scales, and an explainer for the service that produced them. The
rules it shares with every other page are in test_views.py.
"""

from __future__ import annotations

import pytest
from pages import QUESTION, QUESTION_SOURCE, changed

pytestmark = pytest.mark.frontend

#: The view this module is about. `page` in conftest.py reads it.
VIEW = "questions"

#: A question scored since the margins were calibrated, which is what a row
#: written by a current run looks like.
CALIBRATED = changed(
    QUESTION,
    confidence=0.28,
    gate_scores=[
        {
            "gate": "near_duplicate",
            "value": 0.88,
            "threshold": 0.93,
            "margin": 0.2778,
            "high_is_safe": False,
            "safe_end": 0.75,
        },
        {
            "gate": "recall",
            "value": 0.83,
            "threshold": 0.6,
            "margin": 0.575,
            "high_is_safe": True,
            "safe_end": 1.0,
        },
    ],
)


def said(view) -> str:
    """Every caption the page drew, as one string to look in."""
    return " ".join(view.captions())


# ── The explainer ─────────────────────────────────────────────────────────


def test_the_service_explains_itself_in_one_fold(page) -> None:
    """One, not three: a page opening with five collapsed sections floods."""
    folds = [one for one in page().folds() if one.startswith("How ")]

    assert folds == ["How question generation works"]


def test_every_gate_is_listed_whether_or_not_it_fired(page) -> None:
    """A gate missing from a reference is one nobody can look up.

    The page's own list is what the Analysis fold counts against, so a gate
    described here and absent there would be a rejection code with no name.
    """
    listed = page().table_with("Rejected as")

    assert "not_recoverable" in list(listed["Rejected as"])
    assert "source_changed" in list(listed["Rejected as"])


def test_a_gate_says_what_kind_of_thing_it_is(page) -> None:
    """A rule, a measurement and an opinion carry different weight.

    Only a measurement has a margin behind it, which is the whole reason
    the confidence panel is absent on some questions.
    """
    listed = page().table_with("Rejected as")
    kinds = dict(zip(listed["Rejected as"], listed["Kind"], strict=True))

    assert kinds["malformed"] == "rule"
    assert "measurement" in kinds["duplicate"]


def test_the_phase_is_carried_beside_the_code(page) -> None:
    """The two vocabularies reconciled, which is why this table exists.

    A trace records `structural`; the table records `malformed`. Nothing
    told a reader those were the same event until this column did.
    """
    listed = page().table_with("Rejected as")
    phases = dict(zip(listed["Rejected as"], listed["In phase"], strict=True))

    assert phases["malformed"] == "structural"
    assert phases["compound"] == "structural"
    assert phases["duplicate"] == "near_duplicate"


# ── The confidence ────────────────────────────────────────────────────────


def test_the_confidence_names_the_gate_that_was_weakest(page) -> None:
    """The number alone cannot be read: it is the smallest of several.

    Which reading it came from is most of what it says, so a panel showing
    only the figure is showing the least useful half of it.
    """
    opened = page(question=_detail(CALIBRATED), questions=_one(CALIBRATED)).select()

    assert "near_duplicate" in said(opened)
    assert "0.28" in said(opened)


def test_a_reading_says_which_way_its_gate_runs(page) -> None:
    """0.88 is comfortable under one gate and refusing under another."""
    opened = page(question=_detail(CALIBRATED), questions=_one(CALIBRATED)).select()
    table = opened.table_with("Safe when")
    direction = dict(zip(table["Gate"], table["Safe when"], strict=True))

    assert direction["near_duplicate"] == "lower is safer"
    assert direction["recall"] == "higher is safer"


def test_a_legacy_reading_is_not_read_as_the_safe_direction(page) -> None:
    """The fixture carries no direction, as every row written before did.

    Defaulting those to `higher is safer` would tell a reader that a
    question sitting on the duplicate ceiling was as far from refusal as a
    question can get - the exact inversion the column exists to prevent.
    """
    opened = page().select()
    table = opened.table_with("Safe when")
    direction = dict(zip(table["Gate"], table["Safe when"], strict=True))

    assert direction["near_duplicate"] == "lower is safer"


def test_an_uncalibrated_row_says_its_confidence_reads_low(page) -> None:
    """A row scored before the calibration is not a bad question.

    Saying nothing would leave a reader comparing a 0.04 written under one
    arithmetic against a 0.28 written under another.
    """
    opened = page().select()

    assert "calibrated" in said(opened)


def test_a_calibrated_row_carries_no_such_warning(page) -> None:
    """The note is about the row, not about the page."""
    opened = page(question=_detail(CALIBRATED), questions=_one(CALIBRATED)).select()

    assert "before the margins were" not in said(opened)


def test_a_question_nothing_measured_says_so_rather_than_scoring_zero(page) -> None:
    """No confidence is not a low one, and a row of dashes reads as one."""
    unmeasured = changed(QUESTION, confidence=None, gate_scores=[])
    opened = page(question=_detail(unmeasured), questions=_one(unmeasured)).select()

    assert "no confidence" in said(opened)
    assert "not a low one" in said(opened)


def _one(question: dict) -> dict:
    """One page of questions holding exactly this one."""
    return {"total": 1, "questions": [question]}


def _detail(question: dict) -> dict:
    """What the detail route answers for it, which the panel needs whole."""
    return {"question": question, "sources": [QUESTION_SOURCE], "thread": []}
