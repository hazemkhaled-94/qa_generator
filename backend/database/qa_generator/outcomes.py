"""The vocabularies that are not a stage's status.

`StrEnum`, so a member compares equal to a value read back from the database
or arriving in a query parameter.
"""

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """What became of one upload attempt."""

    STORED = "stored"
    DUPLICATE_BYTES = "duplicate_bytes"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"


class QuestionStatus(StrEnum):
    """Where a generated question has got to."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Rejection(StrEnum):
    """Why a candidate fact was not accepted.

    Stored beside the English reason so the quality report groups on a
    stable value rather than on a message carrying a measurement.
    """

    EVIDENCE_ABSENT = "evidence_absent"
    COPIED = "copied"
    NOT_ATOMIC = "not_atomic"
    UNSUPPORTED_ADDITION = "unsupported_addition"
    UNRESOLVED_REFERENCE = "unresolved_reference"


def one_of(column: str, values: type[StrEnum]) -> str:
    """Builds the CHECK expression constraining a column to an enum."""
    return f"{column} IN ({', '.join(f"'{value}'" for value in values)})"
