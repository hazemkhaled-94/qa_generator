"""The base every repository shares."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_

from database.qa_generator.engine import sessions


class Repository:
    """Holds the shared session factory, and nothing else."""

    def __init__(self) -> None:
        """Binds to the session factory."""
        self._session = sessions()


def matching(search: str, *columns: Any) -> Any:
    """Builds the condition a search box's words select over some columns.

    Here rather than in each stage's own filter, because the one thing every
    copy has to remember is `autoescape`: without it a `%` or `_` a person
    typed is a wildcard, and the third copy is where that gets forgotten.
    """
    return or_(*(column.icontains(search, autoescape=True) for column in columns))
