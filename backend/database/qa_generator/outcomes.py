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


class Difficulty(StrEnum):
    """How far a question's evidence is spread.

    Read off the facts a question was written from rather than judged, so
    two people reading the same question agree on it and a report can
    stratify on it without anybody's opinion in the way.
    """

    SINGLE_PASSAGE = "single_passage"
    CROSS_PASSAGE = "cross_passage"
    CROSS_DOCUMENT = "cross_document"


class QuestionRejection(StrEnum):
    """Why a generated question was not accepted.

    Stored in questions.rejected_reason. A code and no message, unlike a
    fact's: no gate here measures anything, so there is nothing a message
    would carry that the code does not.
    """

    MALFORMED = "malformed"
    NOT_RECOVERABLE = "not_recoverable"
    DUPLICATE = "duplicate"
    ANSWERABLE_AFTER_ALL = "answerable_after_all"
    SOURCE_CHANGED = "source_changed"


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
