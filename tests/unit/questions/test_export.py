"""The questions as a workbook.

A spreadsheet is the one artefact here that leaves the deployment and gets
read by somebody who has never seen the pipeline, so what is pinned is what
a reader would notice: that the sheets are there, that a list is a list of
words rather than Python's repr of one, and that nothing raises on the two
inputs that are not a full set - nothing selected, and a cell longer than
Excel will hold.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from question_generation.export import _CELL_LIMIT, _flatten, workbook
from question_generation.models import Citation, QuestionQuality, StoredQuestion


def stored(id: int = 1, **overrides) -> StoredQuestion:
    """One question, as the listing hands it over."""
    fields = {
        "id": id,
        "question_text": "Within how many hours is a request answered?",
        "target_answer": "48 hours",
        "answer_explanation": "A standard request is answered within 48 hours.",
        "answerable": True,
        "difficulty": "easy",
        "passage_scope": "single_passage",
        "document_scope": "single_document",
        "topic_scope": "single_topic",
        "answer_chars": 8,
        "language": "en",
        "status": "accepted",
        "rejected_reason": None,
        "created_at": "2026-09-24T12:00:00+00:00",
        "facts": 1,
        "documents": ["a1b2c3"],
        "topics": ["Support", "Policy"],
    }
    return StoredQuestion(**{**fields, **overrides})


def cited(question_id: int = 1, **overrides) -> Citation:
    """One fact a question rests on."""
    fields = {
        "question_id": question_id,
        "fact_id": 7,
        "statement": "A standard request is answered in 48 hours.",
        "evidence_text": "Requests are answered in 48 hours.",
        "validated": True,
        "document": "Service policy",
        "doc_sha256": "a1b2c3",
        "ordinal": 4,
        "page": 2,
    }
    return Citation(**{**fields, **overrides})


def sheets(drawn: bytes) -> dict[str, pd.DataFrame]:
    """Reads a built workbook back the way a recipient opens it."""
    book = pd.ExcelFile(io.BytesIO(drawn))
    return {name: book.parse(name) for name in book.sheet_names}


def test_the_workbook_has_the_three_sheets() -> None:
    """Questions, what they cite, and what the set came out as."""
    read = sheets(workbook([stored()], [cited()]))

    assert list(read) == ["Questions", "Citations", "Summary"]
    assert len(read["Questions"]) == 1
    assert len(read["Citations"]) == 1


def test_a_question_carries_the_columns_a_reader_filters_on() -> None:
    """The headings are words, and the values are under them."""
    row = sheets(workbook([stored()]))["Questions"].iloc[0]

    assert row["Question"] == "Within how many hours is a request answered?"
    assert row["Answer"] == "48 hours"
    assert row["Type"] is not None
    assert row["Difficulty"] == "easy"
    assert row["Status"] == "accepted"


def test_a_list_is_joined_rather_than_repr_ed() -> None:
    """`topics` is a list, and a cell has no type for one.

    Left to `str()` it would reach a reader as `['Support', 'Policy']`,
    brackets and quotes included.
    """
    row = sheets(workbook([stored()]))["Questions"].iloc[0]

    assert row["Topics"] == "Support, Policy"


def test_nothing_selected_still_has_its_headings() -> None:
    """An empty filter writes an empty sheet, not a broken one.

    A workbook of nothing is a workbook somebody can see the shape of; a
    raise here would mean the one filter that matches no question is the
    one that cannot be exported.
    """
    read = sheets(workbook([], []))

    assert list(read) == ["Questions", "Citations", "Summary"]
    assert read["Questions"].empty
    assert "Question" in read["Questions"].columns


def test_a_cell_too_long_for_excel_is_cut_and_marked() -> None:
    """Excel refuses a cell over 32,767 characters outright.

    Cut with the cut named, so nobody reads a truncated answer as a short
    one - and so a workbook of thousands does not raise on its last row
    after two minutes of work.
    """
    cut = _flatten("x" * (_CELL_LIMIT + 500))

    assert len(cut) < _CELL_LIMIT + 50
    assert cut.endswith("[truncated]")


def test_a_long_answer_survives_the_round_trip() -> None:
    """The cut is at Excel's limit, so an ordinary long answer is untouched."""
    answer = "a sentence. " * 200
    row = sheets(workbook([stored(target_answer=answer)]))["Questions"].iloc[0]

    assert row["Answer"] == answer


@pytest.mark.parametrize("value", [None, 1, True, "plain"])
def test_an_ordinary_value_passes_through(value) -> None:
    """Only lists and over-long text are rewritten."""
    assert _flatten(value) == value


def test_the_summary_counts_what_the_workbook_holds() -> None:
    """Written so a workbook mailed on says what it is.

    Read off the quality report the page was showing rather than counted
    again here, so the figures in the file are the figures on screen.
    """
    quality = QuestionQuality(
        total=4,
        accepted=3,
        draft=0,
        unanswerable=1,
        followups=0,
        topics_covered=1,
        topics_in_coverage=2,
        mean_question_chars=40.0,
        mean_answer_chars=8.0,
        rejected={"duplicate": 1},
        difficulty={"easy": 3, "hard": 1},
        passage_scope={},
        document_scope={},
        topic_scope={},
        question_type={"factoid": 4},
        answer_form={},
        planned_difficulty={},
        cognitive_level={"recall": 4},
        cognitive_level_checked=4,
        planned_met=3,
    )
    summary = sheets(workbook([stored()], [cited()], quality))["Summary"]
    counted = dict(zip(summary["Measure"], summary["Count"], strict=True))

    assert counted["Questions in this workbook"] == 1
    assert counted["Citation rows"] == 1
    assert counted["Matching the filter"] == 4
    assert counted["difficulty: easy"] == 3
    assert counted["rejected: duplicate"] == 1
