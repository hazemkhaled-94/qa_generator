"""Two runs read side by side, live rows and deleted ones alike.

The deleted half is the part worth testing. A rerun deletes the questions it
replaces, so before `archived_rows` existed the baseline was destroyed by
the act of producing the thing to compare it against - which is exactly what
happened to the A/B in `evaluation/README.md`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import insert, text

from database.qa_generator import Question, QuestionStatus
from question_generation.runs import ACCEPTED, RunCatalog, table

pytestmark = pytest.mark.integration


def _question(run: str, reason: str | None, status: str = QuestionStatus.ACCEPTED):
    """One stored question, with only the columns the schema insists on."""
    return {
        "question_text": f"Was kostet das? [{run}/{reason}]",
        "target_answer": None if reason else "vier Euro",
        "answerable": not reason,
        "language": "de",
        "status": status,
        "rejected_reason": reason,
        "run_id": run,
    }


@pytest.fixture
def stored(database, engine):
    """Two runs of questions, the second one refusing more."""
    rows = [
        *[_question("run-a", None) for _ in range(6)],
        *[_question("run-a", "duplicate", QuestionStatus.REJECTED) for _ in range(3)],
        *[_question("run-a", "compound", QuestionStatus.REJECTED) for _ in range(1)],
        *[_question("run-b", None) for _ in range(4)],
        *[_question("run-b", "duplicate", QuestionStatus.REJECTED) for _ in range(1)],
        *[
            _question("run-b", "leaks_source", QuestionStatus.REJECTED)
            for _ in range(5)
        ],
    ]
    with engine.begin() as connection:
        connection.execute(insert(Question), rows)
    return RunCatalog()


def test_each_run_is_counted_on_the_gate_that_stopped_it(stored) -> None:
    """The table the README used to be typed by hand."""
    counted = stored.counts(["run-a", "run-b"])

    assert counted["run-a"].total == 10
    assert counted["run-a"].accepted == 6
    assert counted["run-a"].gates["duplicate"] == 3
    assert counted["run-b"].gates["leaks_source"] == 5
    assert counted["run-b"].share(ACCEPTED) == pytest.approx(0.4)


def test_a_run_that_was_deleted_is_still_compared(stored, engine) -> None:
    """The whole reason this reads the archive.

    `questions-rerun` deletes what it replaces. Run A is gone from
    `questions` and every one of its rows is in `archived_rows`, put there
    by the trigger rather than by anything this knows about.
    """
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM questions WHERE run_id = 'run-a'"))

    counted = stored.counts(["run-a", "run-b"])

    assert counted["run-a"].total == 10, "the deleted run still answers"
    assert counted["run-a"].accepted == 6
    assert counted["run-a"].archived == 10, "and it is reported as deleted"
    assert counted["run-b"].archived == 0


def test_a_draft_counts_as_neither(stored, engine) -> None:
    """A run interrupted mid-topic leaves drafts, which nothing decided."""
    with engine.begin() as connection:
        connection.execute(insert(Question), [_question("run-b", None, "draft")])

    counted = stored.counts(["run-a", "run-b"])

    assert counted["run-b"].total == 10, "the draft is not in the total"
    assert counted["run-b"].accepted == 4


def test_the_table_leads_with_what_moved(stored) -> None:
    """A gate that fired the same in both is not the answer to anything."""
    counted = stored.counts(["run-a", "run-b"])

    lines = table(counted["run-a"], counted["run-b"])
    gates = [line.split()[0] for line in lines[3:] if not line.startswith("note:")]

    assert gates[0] == "leaks_source", "\n".join(lines)
    assert "+50.0pp" in "\n".join(lines)


def test_the_runs_there_are_can_be_listed(stored) -> None:
    """Nobody remembers a uuid, so the ids have to be findable."""
    found = {run: total for run, total, _ in stored.known()}

    assert found == {"run-a": 10, "run-b": 10}
