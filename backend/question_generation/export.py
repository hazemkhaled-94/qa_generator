"""The questions as a spreadsheet, and the command line that writes one.

The dataset had no way out. Every other reader of these rows is a service -
the API pages them, the frontend tables them, Argilla samples them - and a
person who wanted the set itself had psql and nothing else. Argilla is not
that way out either, and it is the near miss worth naming: it holds a
**disposable copy of a stratified sample** pushed for review, so exporting
from it gives back the hundred rows somebody was asked to look at rather
than the set. The database is what has the set.

So: one workbook, three sheets.

- **Questions** - one row per question, the columns a reader filters on,
  and what that question rests on: the facts it cites, the evidence under
  them, its subjects, the documents by name, the answer at length and why
  the judge said what it did. One row is readable on its own, because a
  reader filtering this sheet to the refused questions should not have to
  join another by id to see what any of them cited.
- **Citations** - one row per (question, fact, passage), which is what an
  answer is checked against. A bridge cites two passages and so takes two
  rows. Still here beside the folded-up version above: this is the grain
  that carries the page number and the passage ordinal, which is what
  somebody checking an answer against the source needs.
- **Assessment** - one row per (question, judgement), which is what an
  independent model made of each answer. Empty unless the evaluation phase
  has run.
- **Summary** - what the filter selected, as counts. Written so a workbook
  mailed to somebody says what it is without them having to pivot it.

The judgements arrive as an argument rather than being read here. This
package must not import `assessment` - no backend service imports another,
which `.importlinter` enforces - so the two callers that already hold a
catalogue each look them up and hand them over.

**The filter is the caller's and there is no default scope.** Not "the
release", not "everything accepted": whatever the same filter the listing
and the quality report take says, so the workbook is the thing on screen
and a reader cannot be handed a set narrower than they asked for without
having asked for it.

`openpyxl` and `pandas` are both already dependencies, so this adds none.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from question_generation.models import Citation, QuestionQuality, StoredQuestion

log = logging.getLogger(__name__)

#: The Questions sheet, as (column in the frame, heading in the workbook).
#: Named rather than dumped, because a dataclass field order is not a thing
#: to show somebody and `answer_chars` beside `target_answer` reads as two
#: answers.
_COLUMNS = (
    ("id", "ID"),
    ("question_text", "Question"),
    ("target_answer", "Answer"),
    ("answer_explanation", "Answer in full"),
    ("answerable", "Answerable"),
    ("question_type", "Type"),
    ("answer_form", "Answer form"),
    ("cognitive_level", "Cognitive level"),
    ("difficulty", "Difficulty"),
    ("planned_difficulty", "Difficulty planned"),
    ("passage_scope", "Passage scope"),
    ("document_scope", "Document scope"),
    ("topic_scope", "Topic scope"),
    ("attempt", "Attempt"),
    ("thread_position", "Turn"),
    ("follows_id", "Follows"),
    ("language", "Language"),
    ("status", "Status"),
    ("rejected_reason", "Gate"),
    # How close the weakest measuring gate came to refusing it. A margin
    # and not a probability - see `confidence.py` - and blank on a question
    # no measuring gate read, which is not the same as a low one.
    ("confidence", "Confidence"),
    ("reviewed_verdict", "Reviewed"),
    # Filled from the `judged` argument rather than off the row: a question
    # carries no judgement of its own, and this package cannot read the
    # table that does. Blank where the evaluation phase has not run.
    ("judge", "Judge"),
    ("judge_refused", "Judge refused by"),
    ("judge_why", "Why the judge said so"),
    ("facts", "Facts cited"),
    # Two sets, not pairs: each is de-duplicated on its own, so the nth
    # line of one is not the nth line of the other. The Citations sheet is
    # what puts a fact beside the sentence it was drawn from.
    ("statements", "The facts it cites"),
    ("evidence", "The evidence they rest on"),
    ("topics", "Topics"),
    ("documents", "Documents"),
    ("created_at", "Written"),
)

#: Fields whose list is written one item per line rather than comma-joined.
#: A comma between two sentences reads as one sentence, and both of these
#: hold sentences. The cells are wrapped, so the lines show.
_LINES = frozenset({"statements", "evidence"})

_CITATION_COLUMNS = (
    ("question_id", "Question ID"),
    ("fact_id", "Fact ID"),
    ("statement", "Fact"),
    ("evidence_text", "Evidence"),
    ("validated", "Fact still valid"),
    ("document", "Document"),
    ("page", "Page"),
    ("ordinal", "Passage"),
)

#: The Assessment sheet. One row per judgement rather than per question,
#: because a question is judged three times and a reader wants to sort on
#: which judgement failed.
_ASSESSMENT_COLUMNS = (
    "Question ID",
    "Judgement",
    "Answer",
    "Approved",
    "Score",
    "Why",
    "Judge",
)

#: How wide each heading's column is drawn, by the heading. Anything not
#: named here is sized from its heading, which is right for the short
#: coded columns and wrong for exactly the four long ones.
_WIDTHS = {
    "Question": 60,
    "Answer": 50,
    "Answer in full": 60,
    "Fact": 60,
    "Evidence": 60,
    "Topics": 30,
    "Documents": 30,
    "Why": 70,
    "Judge refused by": 26,
    "Why the judge said so": 70,
    "The facts it cites": 60,
    "The evidence they rest on": 60,
}

#: Columns whose text is wrapped rather than run off under the next cell.
_WRAPPED = frozenset(_WIDTHS)

#: Beyond this, a cell is truncated. Excel refuses a cell over 32,767
#: characters outright, and a workbook that raises at the last row after
#: two minutes of work is worse than one that marks the row.
_CELL_LIMIT = 32_000


def _flatten(value: Any, joiner: str = ", ") -> Any:
    """Renders one field as a cell.

    Lists become a joined string - a spreadsheet has no cell type for a
    list, and `str(list)` would put Python's brackets and quotes in front
    of a reader. Long text is cut at the limit Excel enforces, with the cut
    marked so nobody reads a truncated answer as a short one.

    `joiner` is a comma for the short coded lists and a newline for the two
    holding sentences, which a comma would run together into one.
    """
    if isinstance(value, (list, tuple)):
        value = joiner.join(str(item) for item in value)
    if isinstance(value, str) and len(value) > _CELL_LIMIT:
        return value[:_CELL_LIMIT] + "… [truncated]"
    return value


def _frame(rows: Sequence[Any], columns: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Builds one sheet's frame, in the column order named above."""
    records = [
        {
            heading: _flatten(asdict(row).get(field), "\n" if field in _LINES else ", ")
            for field, heading in columns
        }
        for row in rows
    ]
    # An empty selection still gets its headings, so a workbook of nothing
    # is a workbook a reader can see the shape of rather than a blank page.
    return pd.DataFrame(records, columns=pd.Index([head for _, head in columns]))


def _judged_frame(
    questions: Sequence[StoredQuestion], judged: Mapping[int, Any]
) -> pd.DataFrame:
    """The Questions sheet, with the judge's three columns filled in.

    Filled after the frame is built rather than read off the row, because
    `StoredQuestion` carries no judgement and this package cannot read the
    table that does. The three columns keep their declared position either
    way, which is what building them into `_COLUMNS` as blanks buys.
    """
    frame = _frame(questions, _COLUMNS)
    if not len(frame):
        return frame
    found = [judged.get(one.id) for one in questions]
    frame["Judge"] = [_verdict(one) for one in found]
    frame["Judge refused by"] = [
        ", ".join(getattr(one, "refused", ()) or ()) if one is not None else ""
        for one in found
    ]
    frame["Why the judge said so"] = [_reasoning(one) for one in found]
    return frame


def _reasoning(one: Any) -> str:
    """Every judgement about one question, one per line.

    Beside the verdict rather than only on the Assessment sheet: a reader
    sorting the questions by what the judge refused wants the reason in the
    row they are looking at, and a judgement with no reasoning beside it is
    a label nobody can check.

    The same rendering `review/judged.py` gives a reviewer, written twice
    because it cannot be imported: that module reaches the assessments
    through `assessment.repository`, and no backend service imports
    another.
    """
    if one is None:
        return ""
    return "\n".join(
        f"{metric.metric}: {metric.label} — {metric.explanation}"
        for metric in one.metrics or ()
    )


def _verdict(one: Any) -> str:
    """How one question's judgement reads in a cell.

    `not judged` rather than blank, and the distinction is the point: a
    blank cell in a spreadsheet reads as "no problem found", and a question
    nobody judged is not one the judge approved.
    """
    if one is None or one.approved is None:
        return "not judged"
    return "approved" if one.approved else "refused"


def _assessment_frame(
    questions: Sequence[StoredQuestion], judged: Mapping[int, Any]
) -> pd.DataFrame:
    """The Assessment sheet: one row per judgement of each question.

    Only the questions this workbook holds, so the sheet cannot describe
    rows the filter excluded.
    """
    records = []
    for question in questions:
        one = judged.get(question.id)
        if one is None:
            continue
        for metric in one.metrics or ():
            records.append(
                {
                    "Question ID": question.id,
                    "Judgement": metric.metric,
                    "Answer": metric.label,
                    "Approved": "yes" if metric.approved else "no",
                    "Score": metric.score,
                    "Why": _flatten(metric.explanation),
                    "Judge": one.judge_model,
                }
            )
    return pd.DataFrame(records, columns=pd.Index(list(_ASSESSMENT_COLUMNS)))


def _summary(
    quality: QuestionQuality | None,
    count: int,
    cited: int,
    judged: Mapping[int, Any] | None = None,
) -> pd.DataFrame:
    """The Summary sheet: what this workbook holds, as counts.

    Reads the quality report the API already serves rather than counting
    the frames again, so the figures in the workbook are the figures on the
    page it was exported from.

    The judge's three figures are counted over THIS workbook's questions
    rather than read off the corpus-wide report, because a workbook that
    says "38 disagreements" about a filter holding twelve questions is
    describing something the reader cannot see.
    """
    rows: list[tuple[str, Any]] = [
        ("Questions in this workbook", count),
        ("Citation rows", cited),
    ]
    if judged is not None:
        answered = [one for one in judged.values() if one.approved is not None]
        rows += [
            ("Judged by a model", len(answered)),
            ("Judge approved", sum(1 for one in answered if one.approved)),
            (
                "Kept by the pipeline, refused by the judge",
                sum(1 for one in answered if one.disagrees),
            ),
        ]
    if quality is not None:
        measured = asdict(quality)
        rows += [
            ("Matching the filter", measured.get("total")),
            ("Accepted", measured.get("accepted")),
            ("Unanswerable", measured.get("unanswerable")),
            ("Subjects covered", measured.get("topics_covered")),
            ("Subjects in coverage", measured.get("topics_in_coverage")),
        ]
        for name in ("difficulty", "question_type", "cognitive_level", "rejected"):
            for key, value in sorted((measured.get(name) or {}).items()):
                rows.append((f"{name}: {key}", value))
    return pd.DataFrame(rows, columns=pd.Index(["Measure", "Count"]))


def _dress(sheet: Any, frame: pd.DataFrame) -> None:
    """Freezes the headings, filters them, and sizes the columns.

    Not decoration. A sheet of three thousand rows whose headings scroll
    away is one nobody can read past the first screen, and the four long
    columns run to sixty characters and hide every column after them until
    they are wrapped.
    """
    sheet.freeze_panes = "A2"
    if len(frame):
        sheet.auto_filter.ref = sheet.dimensions
    for index, heading in enumerate(frame.columns, start=1):
        letter = get_column_letter(index)
        sheet.column_dimensions[letter].width = _WIDTHS.get(
            heading, min(max(len(str(heading)) + 2, 10), 24)
        )
        sheet.cell(row=1, column=index).font = Font(bold=True)
        if heading in _WRAPPED:
            for cell in sheet[letter][1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")


def workbook(
    questions: Sequence[StoredQuestion],
    citations: Sequence[Citation] = (),
    quality: QuestionQuality | None = None,
    judged: Mapping[int, Any] | None = None,
) -> bytes:
    """Builds the workbook, and returns it as the bytes of an .xlsx file.

    Bytes rather than a path, because the three callers want different
    things with them: the API streams it, the command line writes it, and
    the frontend hands it to a download button. Nothing here touches a
    filesystem.

    `judged` maps a question's id to what an independent model made of it,
    as `assessment.models.StoredAssessment`. Passed in rather than read
    here, and typed loosely for the same reason: no backend service
    imports another, so this one is handed the verdicts by whichever
    caller already holds a catalogue. None leaves the Assessment sheet out
    and the two Judge columns blank, which is a deployment that has never
    run the evaluation phase.
    """
    held: Mapping[int, Any] = judged or {}
    sheets = {
        "Questions": _judged_frame(questions, held),
        "Citations": _frame(citations, _CITATION_COLUMNS),
        "Assessment": _assessment_frame(questions, held),
        "Summary": _summary(quality, len(questions), len(citations), judged),
    }
    buffer = io.BytesIO()
    # `WriteExcelBuffer` is a protocol a BytesIO satisfies at runtime and
    # the stubs decline to match structurally.
    with pd.ExcelWriter(cast("Any", buffer), engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _dress(writer.sheets[name], frame)
    log.info(
        "wrote a workbook of %d question(s), %d citation row(s) and %d "
        "judgement row(s)",
        len(questions),
        len(citations),
        len(sheets["Assessment"]),
    )
    return buffer.getvalue()


def parser() -> argparse.ArgumentParser:
    """Builds the export command line.

    Every filter is the one the listing route takes, spelled the same way,
    so a workbook can be asked for with what a page was narrowed by.
    """
    built = argparse.ArgumentParser(
        prog="python -m question_generation.export",
        description="Write the questions a filter selects to an .xlsx file.",
    )
    built.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="where to write the workbook",
    )
    built.add_argument("--status", choices=("draft", "accepted", "rejected"))
    built.add_argument("--document", metavar="SHA256", help="one document's sha256")
    built.add_argument("--topic", type=int, metavar="ID")
    built.add_argument("--question-type", dest="question_type", metavar="KIND")
    built.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    built.add_argument(
        "--planned-difficulty",
        dest="planned_difficulty",
        choices=("easy", "medium", "hard"),
        help="the band the plan asked for, as against the one it came out",
    )
    built.add_argument(
        "--cognitive-level",
        dest="cognitive_level",
        choices=("recall", "understand", "apply", "analyse"),
    )
    built.add_argument("--answer-form", dest="answer_form", metavar="FORM")
    built.add_argument(
        "--passage-scope",
        dest="passage_scope",
        choices=("single_passage", "multi_passage"),
    )
    built.add_argument(
        "--document-scope",
        dest="document_scope",
        choices=("single_document", "cross_document"),
    )
    built.add_argument(
        "--topic-scope", dest="topic_scope", choices=("single_topic", "multi_topic")
    )
    built.add_argument("--search", metavar="TEXT", help="what the search box takes")
    built.add_argument(
        "--field", choices=("question", "answer", "both"), default="both"
    )
    answerable = built.add_mutually_exclusive_group()
    answerable.add_argument(
        "--answerable", action="store_true", help="only the ones with an answer"
    )
    answerable.add_argument(
        "--unanswerable", action="store_true", help="only the ones without"
    )
    # A follow-up cannot be read without the turn before it, so an exam set
    # wants `--roots-only` and a conversational benchmark wants the threads.
    follows = built.add_mutually_exclusive_group()
    follows.add_argument(
        "--roots-only",
        action="store_true",
        help="drop the follow-ups, which cannot be asked on their own",
    )
    follows.add_argument(
        "--followups-only", action="store_true", help="only the follow-ups"
    )
    built.add_argument(
        "--no-citations",
        action="store_true",
        help="leave the Citations sheet empty, which is much faster on a "
        "whole corpus and enough when only the questions are wanted",
    )
    return built


def main(argv: list[str] | None = None) -> int:
    """Writes one workbook from the command line.

    Returns:
        The process exit code.
    """
    import telemetry
    from assessment.repository import AssessmentCatalog
    from database.qa_generator import engine
    from question_generation.catalog import QuestionCatalog

    args = parser().parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("questions-export")
    telemetry.trace_engine(engine())

    answerable = True if args.answerable else (False if args.unanswerable else None)
    follows = True if args.followups_only else (False if args.roots_only else None)
    where: dict[str, Any] = {
        "document": args.document,
        "topic": args.topic,
        "search": args.search,
        "field": args.field,
        "status": args.status,
        "answerable": answerable,
        "follows": follows,
        "question_type": args.question_type,
        "difficulty": args.difficulty,
        "planned_difficulty": args.planned_difficulty,
        "cognitive_level": args.cognitive_level,
        "answer_form": args.answer_form,
        "passage_scope": args.passage_scope,
        "document_scope": args.document_scope,
        "topic_scope": args.topic_scope,
    }

    catalog = QuestionCatalog()
    # The listing pages, and an export is every page. Unlimited rather than
    # counted and then windowed to that count: those were two statements
    # with a gap between them, and a question written into the gap did not
    # fit the window the count had already decided on.
    _, questions = catalog.page(limit=None, **where)
    citations = () if args.no_citations else catalog.citations(**where)

    # Imported inside `main` rather than at the top of the module, like
    # the catalogue beside it. `workbook` is also called by the api, and
    # no backend service may import another - this is the command line
    # rather than the library, and it is allowed to know about both.
    judged = AssessmentCatalog().verdicts_for("question", [one.id for one in questions])

    with open(args.out, "wb") as handle:
        handle.write(workbook(questions, citations, catalog.quality(**where), judged))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
