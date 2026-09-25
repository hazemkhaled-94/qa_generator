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
        "documents": ["Service policy"],
        "topics": ["Support", "Policy"],
        "statements": ["A standard request is answered in 48 hours."],
        "evidence": ["Requests are answered in 48 hours."],
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


def test_the_workbook_has_the_four_sheets() -> None:
    """Questions, what they cite, what a judge said, and the set's shape."""
    read = sheets(workbook([stored()], [cited()]))

    assert list(read) == ["Questions", "Citations", "Assessment", "Summary"]
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


def test_a_question_row_carries_what_it_rests_on() -> None:
    """The facts, the evidence and the reason, on the question's own row.

    The Citations sheet holds the same pairs one row apiece, and a reader
    who has filtered the Questions sheet to the refused ones should not
    have to join it by id to see what any of them cited.
    """
    row = sheets(workbook([stored()]))["Questions"].iloc[0]

    assert row["The facts it cites"] == "A standard request is answered in 48 hours."
    assert row["The evidence they rest on"] == "Requests are answered in 48 hours."
    assert row["Answer in full"] == "A standard request is answered within 48 hours."
    # The title, not the digest: a sha256 is not a thing anybody can look
    # up by hand, which is why the Citations sheet already resolves one.
    assert row["Documents"] == "Service policy"


def test_sentences_are_one_per_line_rather_than_comma_joined() -> None:
    """A comma between two sentences reads as one sentence."""
    cited = ["The first holds.", "The second holds."]
    row = sheets(workbook([stored(statements=cited)]))["Questions"].iloc[0]

    assert row["The facts it cites"] == "The first holds.\nThe second holds."


class _Judged:
    """One question's judgement, as the assessment catalogue hands it over."""

    def __init__(self, *metrics) -> None:
        """Holds these judgements and nothing else."""
        self.metrics = metrics
        self.approved = all(one.approved for one in metrics)
        self.judge_model = "a-model"
        self.refused = [one.metric for one in metrics if not one.approved]
        self.disagrees = False


class _Metric:
    """One metric of one judgement."""

    def __init__(self, metric: str, label: str, approved: bool, why: str) -> None:
        """Holds what the judge said about this one metric."""
        self.metric, self.label, self.approved, self.explanation = (
            metric,
            label,
            approved,
            why,
        )
        self.score = 1.0 if approved else 0.0


def test_the_judges_reasoning_is_on_the_question_row() -> None:
    """Not only on the Assessment sheet.

    A reader sorting the questions by what the judge refused wants the
    reason in the row they are looking at; a label with no reasoning
    beside it is one nobody can check.
    """
    refused = _Metric("groundedness", "unsupported", False, "The passages say 24.")
    row = sheets(workbook([stored()], judged={1: _Judged(refused)}))["Questions"].iloc[
        0
    ]

    assert row["Judge"] == "refused"
    assert row["Judge refused by"] == "groundedness"
    assert "The passages say 24." in row["Why the judge said so"]


def test_an_unjudged_question_says_so_rather_than_going_blank() -> None:
    """A blank cell reads as "no problem found"."""
    row = sheets(workbook([stored()]))["Questions"].iloc[0]

    assert row["Judge"] == "not judged"
    assert pd.isna(row["Why the judge said so"])


def test_nothing_selected_still_has_its_headings() -> None:
    """An empty filter writes an empty sheet, not a broken one.

    A workbook of nothing is a workbook somebody can see the shape of; a
    raise here would mean the one filter that matches no question is the
    one that cannot be exported.
    """
    read = sheets(workbook([], []))

    assert list(read) == ["Questions", "Citations", "Assessment", "Summary"]
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


class _Catalog:
    """A catalogue that records how the export asked for its rows."""

    def __init__(self, rows: int = 3) -> None:
        """Holds that many questions."""
        self._rows = rows
        self.limits: list[int | None] = []

    def page(self, limit=50, **where):
        """Records the window asked for and answers within it."""
        del where
        self.limits.append(limit)
        taken = self._rows if limit is None else min(limit, self._rows)
        return self._rows, [stored(one) for one in range(1, taken + 1)]

    def citations(self, **where):
        """No citations; the sheet is optional and off here."""
        del where
        return []

    def quality(self, **where):
        """No report, which the workbook takes as None."""
        del where


class _NoJudgements:
    """An assessment catalogue for a corpus nobody has judged.

    Which is every corpus until the evaluation phase is switched on, so
    this is the ordinary case rather than a stub for an unusual one.
    """

    def verdicts_for(self, kind: str, ids: list[int]) -> dict:
        """Nothing has been judged."""
        del kind, ids
        return {}


def test_the_export_asks_for_every_row_in_one_window(tmp_path, monkeypatch) -> None:
    """It counted, then asked for a window the size of the count.

    Two statements with a gap between them: a question written into the
    gap did not fit the window the count had already decided on, so the
    workbook came up a row short of the figure on its own Summary sheet.

    Patched where `main` imports from rather than on the module: those
    three imports are inside the function, so the name this binds is the
    one it looks up when it runs.
    """
    from question_generation import export as module

    catalog = _Catalog(rows=3)
    monkeypatch.setattr("question_generation.catalog.QuestionCatalog", lambda: catalog)
    monkeypatch.setattr("database.qa_generator.engine", lambda: None)
    monkeypatch.setattr("telemetry.trace_engine", lambda engine: None)
    # The judge's verdicts are looked up the same way the questions are,
    # and are the same kind of import inside `main`. Patched here so this
    # stays a test about the window and not about a database.
    monkeypatch.setattr(
        "assessment.repository.AssessmentCatalog", lambda: _NoJudgements()
    )
    written = tmp_path / "questions.xlsx"

    module.main(["--out", str(written), "--no-citations"])

    assert catalog.limits == [None], "one unlimited read, not a count and a window"
    assert written.exists()
