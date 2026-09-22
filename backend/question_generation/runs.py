"""Two runs, read side by side on the gate that stopped each question.

The tables in `evaluation/README.md` were typed by hand off two terminals,
which is why there are only a few of them and why the A/B had to delete the
first run's questions before starting the second - there was no other way to
tell the two apart afterwards. `questions.run_id` is what makes this a
query; see `settings/runs.py` for what names a run.

**Live rows and archived ones together, and that is the point.** A rerun
deletes the questions it replaces, and before the archive existed those rows
were gone: run A could not be compared against run B because starting B was
what destroyed A. `archived_rows` keeps the row as it was, `run_id` is in
the payload with everything else, and a run deleted an hour ago still
answers here.

Reporting only. Nothing here writes, and nothing gates on what it prints:
a model's answers move between two runs at the same temperature, which is
the reason `evaluation/` never gates either.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import func, literal, select, union_all

from database.qa_generator import Question, QuestionStatus
from database.qa_generator.archived_rows import ArchivedRow
from database.qa_generator.repository import Repository

log = logging.getLogger(__name__)

#: What an accepted question is grouped under. `rejected_reason` is NULL for
#: one, and NULL is not a group a reader can read.
ACCEPTED = "accepted"

#: How wide the gate column is printed. The longest code is
#: `answerable_after_all` at twenty.
_GATE = 24


@dataclass(frozen=True)
class RunCounts:
    """One run, as how many questions each gate stopped.

    Attributes:
        run: What named the run.
        gates: How many questions carry each rejection code, with the
            accepted ones under `accepted`.
        archived: How many of the rows counted came out of the archive
            rather than out of `questions`. Reported because a run that is
            entirely archived has been deleted, and a reader comparing
            against it should know that before trusting the numbers.
    """

    run: str
    gates: Counter = field(default_factory=Counter)
    archived: int = 0

    @property
    def total(self) -> int:
        """How many questions the run wrote."""
        return sum(self.gates.values())

    @property
    def accepted(self) -> int:
        """How many cleared every gate."""
        return self.gates.get(ACCEPTED, 0)

    def share(self, gate: str) -> float:
        """What share of the run one gate accounts for, in [0, 1]."""
        return self.gates.get(gate, 0) / self.total if self.total else 0.0


class RunCatalog(Repository):
    """Reads what became of the questions one run wrote."""

    def _counted(self, runs: list[str]):
        """Live questions and archived ones, as one set of grouped counts.

        Two selects rather than a join: the live row has columns and the
        archived one has a JSON payload, so the shapes only meet once both
        are reduced to (run, gate, archived, count).
        """
        live = select(
            Question.run_id.label("run"),
            func.coalesce(Question.rejected_reason, ACCEPTED).label("gate"),
            literal(False).label("archived"),
            func.count().label("total"),
        ).where(
            Question.run_id.in_(runs),
            # A draft is a question the run had not finished with. Counting
            # it as accepted would overstate every run that was interrupted.
            Question.status != QuestionStatus.DRAFT,
        )

        gate = func.coalesce(ArchivedRow.payload["rejected_reason"].astext, ACCEPTED)
        archived = select(
            ArchivedRow.payload["run_id"].astext.label("run"),
            gate.label("gate"),
            literal(True).label("archived"),
            func.count().label("total"),
        ).where(
            ArchivedRow.table_name == "questions",
            ArchivedRow.payload["run_id"].astext.in_(runs),
            ArchivedRow.payload["status"].astext != QuestionStatus.DRAFT,
        )

        return union_all(
            live.group_by("run", "gate", "archived"),
            archived.group_by("run", "gate", "archived"),
        )

    def counts(self, runs: list[str]) -> dict[str, RunCounts]:
        """What each named run wrote, by the gate that stopped it."""
        gates: dict[str, Counter] = {run: Counter() for run in runs}
        archived: Counter = Counter()
        with self._session() as session:
            for run, gate, was_archived, total in session.execute(self._counted(runs)):
                if run is None or run not in gates:
                    continue
                gates[run][gate] += total
                if was_archived:
                    archived[run] += total
        return {
            run: RunCounts(run=run, gates=gates[run], archived=archived[run])
            for run in runs
        }

    def known(self, limit: int = 20) -> list[tuple[str, int, bool]]:
        """The runs there are, newest first, as (run, questions, archived).

        So a reader can find the two to compare without knowing a uuid by
        heart. Newest by the highest question id the run wrote, which is the
        order they happened in and needs no timestamp.
        """
        live = select(
            Question.run_id.label("run"),
            func.count().label("total"),
            func.max(Question.id).label("newest"),
            literal(False).label("archived"),
        ).where(Question.run_id.is_not(None))
        archived = select(
            ArchivedRow.payload["run_id"].astext.label("run"),
            func.count().label("total"),
            func.max(ArchivedRow.id).label("newest"),
            literal(True).label("archived"),
        ).where(
            ArchivedRow.table_name == "questions",
            ArchivedRow.payload["run_id"].astext.is_not(None),
        )
        combined = union_all(
            live.group_by("run", "archived"), archived.group_by("run", "archived")
        ).subquery()

        rolled = (
            select(
                combined.c.run,
                func.sum(combined.c.total).label("total"),
                func.max(combined.c.newest).label("newest"),
                func.bool_and(combined.c.archived).label("gone"),
            )
            .group_by(combined.c.run)
            .order_by(func.max(combined.c.newest).desc())
            .limit(limit)
        )
        with self._session() as session:
            return [
                (run, total, gone) for run, total, _, gone in session.execute(rolled)
            ]


def table(left: RunCounts, right: RunCounts) -> list[str]:
    """The two runs as one table, every gate either of them fired.

    Returned rather than printed, so what a person reads is what a test
    reads - the shape `settings/run.py` uses.

    Ordered by how much the two DISAGREE, largest first, because that is
    the question being asked. A gate that fired the same number of times in
    both is the answer "nothing changed here" and belongs at the bottom.
    """
    gates = set(left.gates) | set(right.gates)
    moved = sorted(
        gates, key=lambda gate: (-abs(left.share(gate) - right.share(gate)), gate)
    )
    lines = [
        f"{'':<{_GATE}} {left.run[:18]:>18} {right.run[:18]:>18}   change",
        f"{'total written':<{_GATE}} {left.total:>18} {right.total:>18}",
        f"{'':-<{_GATE + 60}}",
    ]
    for gate in moved:
        first, second = left.gates.get(gate, 0), right.gates.get(gate, 0)
        change = (right.share(gate) - left.share(gate)) * 100
        lines.append(
            f"{gate:<{_GATE}} "
            f"{first:>10} {left.share(gate):>7.1%} "
            f"{second:>10} {right.share(gate):>7.1%} "
            f"{change:>+8.1f}pp"
        )
    for side in (left, right):
        if side.archived:
            lines.append(
                f"note: {side.archived} of {side.run}'s {side.total} rows are "
                f"archived, so that run has been deleted and is being read "
                f"out of archived_rows"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Runs the comparison command line."""
    import telemetry
    from database.qa_generator import engine

    built = argparse.ArgumentParser(prog="python -m question_generation.runs")
    built.add_argument(
        "runs",
        nargs="*",
        metavar="RUN",
        help="two runs to compare, oldest first. None lists what there is.",
    )
    built.add_argument(
        "--list",
        action="store_true",
        help="list the runs there are, newest first, and stop",
    )
    args = built.parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("questions")
    telemetry.trace_engine(engine())
    catalog = RunCatalog()

    if args.list or not args.runs:
        found = catalog.known()
        if not found:
            log.info(
                "no run has been named yet. Every run from this commit "
                "forward carries one; the rows written before it have "
                "run_id NULL and cannot be told apart."
            )
            return 0
        log.info("%-34s %9s", "run", "questions")
        for run, total, gone in found:
            log.info("%-34s %9d%s", run, total, "  (deleted)" if gone else "")
        return 0

    if len(args.runs) != 2:
        raise SystemExit(
            f"two runs are compared, and {len(args.runs)} were named. "
            f"`--list` shows what there is."
        )

    left, right = args.runs
    counted = catalog.counts([left, right])
    for line in table(counted[left], counted[right]):
        log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
