"""The condition a search box's words select.

`autoescape` is the thing every copy of this has to remember: without it a
`%` or `_` a person typed is a wildcard.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from database.qa_generator.documents import Document
from database.qa_generator.repository import matching


def compiled(search: str, *columns):
    """Compiles the statement one search would run, with its parameters."""
    return (
        select(Document.sha256)
        .where(matching(search, *columns))
        .compile(dialect=postgresql.dialect())
    )


def test_a_wildcard_a_person_typed_is_escaped() -> None:
    """`50%` searches for the characters, not for anything at all."""
    statement = compiled("50%", Document.title)

    assert "ESCAPE" in str(statement)
    assert "50%" not in "".join(str(v) for v in statement.params.values())


def test_an_underscore_is_escaped_too() -> None:
    """`_` matches one character unless it is escaped."""
    statement = compiled("a_b", Document.title)

    assert "ESCAPE" in str(statement)
    assert "a_b" not in "".join(str(v) for v in statement.params.values())


def test_an_ordinary_word_survives_intact() -> None:
    """Escaping costs nothing when there is nothing to escape."""
    statement = compiled("bafin", Document.title)

    assert any("bafin" in str(v) for v in statement.params.values())


def test_every_column_given_is_searched() -> None:
    """One search box, several columns, joined by OR."""
    statement = str(compiled("bafin", Document.title, Document.sha256))

    assert " OR " in statement
    assert "title" in statement
    assert "sha256" in statement


def test_one_column_needs_no_or() -> None:
    """A single column is the condition itself."""
    assert " OR " not in str(compiled("bafin", Document.title))
