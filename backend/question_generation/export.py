"""The questions as a spreadsheet, and the command line that writes one.

The dataset had no way out. Every other reader of these rows is a service -
the API pages them, the frontend tables them, Argilla samples them - and a
person who wanted the set itself had psql and nothing else. Argilla is not
that way out either, and it is the near miss worth naming: it holds a
**disposable copy of a stratified sample** pushed for review, so exporting
from it gives back the hundred rows somebody was asked to look at rather
than the set. The database is what has the set.

So: one workbook, three sheets.

- **Questions** - one row per question, the columns a reader filters on.
- **Citations** - one row per (question, fact, passage), which is what an
  answer is checked against. A bridge cites two passages and so takes two
  rows.
- **Summary** - what the filter selected, as counts. Written so a workbook
  mailed to somebody says what it is without them having to pivot it.

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
from collections.abc import Sequence
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
    ("thread_position", "Turn"),
    ("follows_id", "Follows"),
    ("language", "Language"),
    ("status", "Status"),
    ("rejected_reason", "Gate"),
    ("reviewed_verdict", "Reviewed"),
    ("facts", "Facts cited"),
    ("topics", "Topics"),
    ("documents", "Documents"),
    ("created_at", "Written"),
)

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
}

#: Columns whose text is wrapped rather than run off under the next cell.
_WRAPPED = frozenset(_WIDTHS)

#: Beyond this, a cell is truncated. Excel refuses a cell over 32,767
#: characters outright, and a workbook that raises at the last row after
#: two minutes of work is worse than one that marks the row.
_CELL_LIMIT = 32_000


def _flatten(value: Any) -> Any:
    """Renders one field as a cell.

    Lists become a comma-joined string - a spreadsheet has no cell type for
    a list, and `str(list)` would put Python's brackets and quotes in front
    of a reader. Long text is cut at the limit Excel enforces, with the cut
    marked so nobody reads a truncated answer as a short one.
    """
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    if isinstance(value, str) and len(value) > _CELL_LIMIT:
        return value[:_CELL_LIMIT] + "… [truncated]"
    return value


def _frame(rows: Sequence[Any], columns: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Builds one sheet's frame, in the column order named above."""
    records = [
        {heading: _flatten(asdict(row).get(field)) for field, heading in columns}
        for row in rows
    ]
    # An empty selection still gets its headings, so a workbook of nothing
    # is a workbook a reader can see the shape of rather than a blank page.
    return pd.DataFrame(records, columns=pd.Index([head for _, head in columns]))


def _summary(quality: QuestionQuality | None, count: int, cited: int) -> pd.DataFrame:
    """The Summary sheet: what this workbook holds, as counts.

    Reads the quality report the API already serves rather than counting
    the frames again, so the figures in the workbook are the figures on the
    page it was exported from.
    """
    rows: list[tuple[str, Any]] = [
        ("Questions in this workbook", count),
        ("Citation rows", cited),
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
) -> bytes:
    """Builds the workbook, and returns it as the bytes of an .xlsx file.

    Bytes rather than a path, because the three callers want different
    things with them: the API streams it, the command line writes it, and
    the frontend hands it to a download button. Nothing here touches a
    filesystem.
    """
    sheets = {
        "Questions": _frame(questions, _COLUMNS),
        "Citations": _frame(citations, _CITATION_COLUMNS),
        "Summary": _summary(quality, len(questions), len(citations)),
    }
    buffer = io.BytesIO()
    # `WriteExcelBuffer` is a protocol a BytesIO satisfies at runtime and
    # the stubs decline to match structurally.
    with pd.ExcelWriter(cast("Any", buffer), engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _dress(writer.sheets[name], frame)
    log.info(
        "wrote a workbook of %d question(s) and %d citation row(s)",
        len(questions),
        len(citations),
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

    with open(args.out, "wb") as handle:
        handle.write(workbook(questions, citations, catalog.quality(**where)))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
