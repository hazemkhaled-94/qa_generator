"""The states a stage can put a row in."""

from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    """One stage's view of a row.

    A row arrives `new`, which no worker looks at; starting the stage moves
    it to `pending`, which is the only status a worker claims. That is what
    keeps one stage from setting another going.
    """

    NEW = "new"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EXTRACTED = "extracted"
    MODELLED = "modelled"
    GENERATED = "generated"
    ASSESSED = "assessed"


#: The states shared by every stage. A stage's own set is these plus its done
#: value.
SHARED = (Status.NEW, Status.PENDING, Status.IN_PROGRESS, Status.FAILED)

#: The states a worker still has to act on, which is what a queue index covers.
OUTSTANDING = (Status.PENDING, Status.IN_PROGRESS)


def check(column: str, done: Status) -> str:
    """Builds the CHECK expression constraining one stage's status column."""
    allowed = ", ".join(f"'{status}'" for status in (*SHARED, done))
    return f"{column} IN ({allowed})"


def queued(column: str) -> str:
    """Builds the predicate a stage's partial queue index covers."""
    return f"{column} IN ({', '.join(f"'{status}'" for status in OUTSTANDING)})"
