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


class PassageScope(StrEnum):
    """Whether one passage answers the question or several are needed."""

    SINGLE = "single_passage"
    MULTI = "multi_passage"


class DocumentScope(StrEnum):
    """Whether the answer sits in one document or spans two.

    The one a retriever cannot fake: a cross-document question has no single
    chunk that contains its answer.
    """

    SINGLE = "single_document"
    CROSS = "cross_document"


class TopicScope(StrEnum):
    """Whether the question stays inside one subject or bridges two.

    A passage usually belongs to several topics above the weight floor, so a
    question drawing on two subjects is a question about the material rather
    than about a section of it.
    """

    SINGLE = "single_topic"
    MULTI = "multi_topic"


class Difficulty(StrEnum):
    """How hard a question is to answer, as a band.

    Derived and not judged: the three scopes above, the length of the answer
    and whether the question follows another are each worth a point, and the
    band is the total. Two people reading the same question agree on it, and
    a review sample can stratify on it without anybody's opinion in the way.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionRejection(StrEnum):
    """Why a generated question was not accepted.

    Stored in questions.rejected_reason. A code and no message, unlike a
    fact's: no gate here measures anything, so there is nothing a message
    would carry that the code does not.
    """

    MALFORMED = "malformed"
    ANSWER_TOO_SHORT = "answer_too_short"
    RESTATES_FACT = "restates_fact"
    UNANCHORED = "unanchored"
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
