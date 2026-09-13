"""Narrowing a queue operation to one item.

The condition is built in one place and executed in another, so a narrowing
that never reaches the SQL takes the whole queue and reports a success.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.dialects import postgresql

from database.qa_generator import Status
from extraction.repository import PassageQueue
from preprocessing.parsing.repository import ParseQueue
from stages.queue import _also


def where(queue: Any, within: Any) -> str:
    """Compiles the WHERE one operation would run with."""
    return str(
        update(queue.columns.entity)
        .where(*_also(queue.columns.status == Status.PENDING, within))
        .values({queue.columns.status: Status.NEW})
        .compile(dialect=postgresql.dialect())
    )


def test_a_narrowing_nobody_gave_drops_out() -> None:
    """`where()` takes no None."""
    assert _also(None) == ()
    assert _also("a", None, "b") == ("a", "b")


def test_a_narrowing_reaches_the_sql_beside_the_status() -> None:
    """Parsing narrowed to a document selects on its own key."""
    narrowed = where(ParseQueue(), ParseQueue().narrow("document", "abc"))
    assert "sha256" in narrowed, narrowed
    assert "parse_status" in narrowed, narrowed


def test_the_whole_queue_form_carries_no_narrowing() -> None:
    """The same method, one argument apart."""
    parsing = ParseQueue()
    assert "sha256" not in where(parsing, None), where(parsing, None)


def test_extraction_narrowed_to_a_document_selects_the_passage_document() -> None:
    """Extraction queues over passages and is asked for a document."""
    extraction = PassageQueue()
    by_document = where(extraction, extraction.narrow("document", "abc"))
    assert "doc_sha256" in by_document, by_document
    assert "passages.id" not in by_document, by_document


def test_a_value_is_coerced_to_what_the_column_holds() -> None:
    """An integer column compared against a string is an error PostgreSQL raises."""
    assert PassageQueue().narrow("passage", "7").right.value == 7


@pytest.mark.parametrize(
    ("queue", "scope", "value", "expected"),
    [
        (ParseQueue, "passage", "1", KeyError),
        (PassageQueue, "document", "abc", None),
        (PassageQueue, "passage", "not-a-number", ValueError),
        (PassageQueue, "nonsense", "1", KeyError),
    ],
)
def test_a_scope_a_stage_does_not_take_is_refused(
    queue, scope, value, expected
) -> None:
    """Each stage accepts its own scopes, and each scope its own values."""
    if expected is None:
        assert queue().narrow(scope, value) is not None
        return
    with pytest.raises(expected):
        queue().narrow(scope, value)
