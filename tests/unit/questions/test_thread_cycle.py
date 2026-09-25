"""A thread that closes into a loop is read once, not for ever.

`questions.follows_id` is a self-referential foreign key and nothing in the
schema stops it closing: `thread_position` counts up, and no constraint
compares a row's to its parent's. A generation run never writes one - the
service links each turn to the row it has just inserted - but a backfill, a
restore out of `archived_rows` or a statement typed into Adminer can, and
the first page to read that question hung rather than showing something
wrong.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from question_generation.catalog import QuestionCatalog
from question_generation.models import StoredQuestion

#: How many reads a walk may make before the test calls it runaway. A hang
#: in a test is a timeout with nothing to read; this is a failure that says
#: which walk did not terminate.
_RUNAWAY = 50


def question(question_id: int, follows_id: int | None) -> StoredQuestion:
    """One row, carrying only what `_thread` reads off it."""
    return StoredQuestion(
        id=question_id,
        question_text="How long is allowed for a standard request?",
        target_answer="48 hours",
        answer_explanation=None,
        answerable=True,
        difficulty="easy",
        passage_scope="single_passage",
        document_scope="single_document",
        topic_scope="single_topic",
        answer_chars=8,
        language="en",
        status="accepted",
        rejected_reason=None,
        created_at=None,
        facts=1,
        documents=[],
        topics=[],
        follows_id=follows_id,
    )


class Threaded(QuestionCatalog):
    """A catalogue over rows held in memory, with no database behind it.

    `Repository.__init__` is deliberately not called: it binds the session
    factory, and all `_thread` asks a session for is one scalar.
    """

    def __init__(self, follows: dict[int, int | None]) -> None:
        """Takes each question's parent, by id."""
        self._follows = follows
        self._children = {
            parent: child for child, parent in follows.items() if parent is not None
        }
        self.reads = 0

    def one(self, question_id: int) -> StoredQuestion | None:
        """The question with that id, as the listing would show it."""
        if question_id not in self._follows:
            return None
        self.reads += 1
        assert self.reads < _RUNAWAY, "the walk did not terminate"
        return question(question_id, self._follows[question_id])

    @property
    def _session(self):
        """A session whose only answer is the next turn of the thread."""

        @contextmanager
        def opened():
            """Hands back something that answers `scalar`."""
            yield _Answers(self._children, self)

        return opened


class _Answers:
    """Answers `select(id).where(follows_id == n)` off a dict."""

    def __init__(self, children: dict[int, int], catalog: Threaded) -> None:
        """Takes each question's follow-up, by parent id."""
        self._children = children
        self._catalog = catalog

    def scalar(self, statement):
        """The child of whichever id the statement compares against.

        Read off the compiled parameters rather than out of the clause
        tree: that is the shape SQLAlchemy documents, and it does not move
        when the expression is built a different way.
        """
        self._catalog.reads += 1
        assert self._catalog.reads < _RUNAWAY, "the walk did not terminate"
        (parent,) = statement.compile().params.values()
        return self._children.get(parent)


def test_a_straight_thread_is_read_from_its_root() -> None:
    """The ordinary case, and what the guard must not change."""
    catalog = Threaded({1: None, 2: 1, 3: 2})

    walked = catalog._thread(question(2, 1))

    assert [one.id for one in walked] == [1, 2, 3]


def test_a_question_that_follows_itself_terminates() -> None:
    """The walk up never left it."""
    catalog = Threaded({1: 1})

    assert [one.id for one in catalog._thread(question(1, 1))] == [1]


def test_a_thread_that_closes_into_a_loop_terminates() -> None:
    """Two rows each following the other: a page that never loads."""
    catalog = Threaded({1: 2, 2: 1})

    walked = catalog._thread(question(1, 2))

    assert {one.id for one in walked} <= {1, 2}


@pytest.mark.parametrize(
    "follows", [{1: None, 2: 1, 3: 2}, {1: 3, 2: 1, 3: 2}, {1: 1}, {1: 2, 2: 1}]
)
def test_every_shape_is_read_in_bounded_time(follows) -> None:
    """Straight or closed, no row is read more than a couple of times."""
    catalog = Threaded(follows)

    catalog._thread(question(1, follows[1]))

    assert catalog.reads <= 2 * len(follows) + 2
