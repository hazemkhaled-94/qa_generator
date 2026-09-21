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


class QuestionType(StrEnum):
    """What kind of thing a question asks for.

    Question form, not subject matter: every one of these is askable of any
    corpus. QUESTIONS_TYPE_MIX says which are written and in what proportion.
    """

    FACTOID = "factoid"
    DEFINITION = "definition"
    ENTITY = "entity"
    ENUMERATION = "enumeration"
    CONDITION = "condition"
    REASON = "reason"
    PROCEDURE = "procedure"
    CONSEQUENCE = "consequence"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    IMPLICATION = "implication"
    APPLICATION = "application"


class CognitiveLevel(StrEnum):
    """How much a question asks of whoever answers it.

    Derived from the question's type rather than judged, the way
    `difficulty` is derived from its scopes: a type declares what it asks
    for, and what it asks for decides this. Nobody holds an opinion about
    an individual row.

    It is a different axis from `difficulty`, and the two are deliberately
    not merged. Difficulty says how much of the corpus an answer is spread
    over - how hard it is to FIND - and a question spanning two documents
    can be a bare lookup once both are in hand. This says how much has to
    be done with what was found.

    RECALL     the answer is a value stated in one place
    UNDERSTAND the answer restates what the material means
    APPLY      the answer maps a rule the material gives onto a case
    ANALYSE    the answer is not stated anywhere and has to be worked out
    """

    RECALL = "recall"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYSE = "analyse"


class Derivation(StrEnum):
    """How an answer that is not stated is got out of the material.

    Which gate a question faces. Recoverability asks whether the passages
    STATE the answer, and for these types that question is the wrong one -
    their whole point is that the answer is not there to be found.

    ARITHMETIC the figures are stated and the total is not
    ENTAILMENT the premises are stated and the conclusion is not
    """

    ARITHMETIC = "arithmetic"
    ENTAILMENT = "entailment"


class AnswerForm(StrEnum):
    """The shape the target answer takes.

    Each question type declares one. The gates read it: a value may carry no
    verb, an explanation must, and each has its own length bounds.
    """

    VALUE = "value"
    LIST = "list"
    EXPLANATION = "explanation"


class QuestionRejection(StrEnum):
    """Why a generated question was not accepted.

    Stored in questions.rejected_reason. A code and no message, unlike a
    fact's: no gate here measures anything, so there is nothing a message
    would carry that the code does not.
    """

    MALFORMED = "malformed"
    COMPOUND = "compound"
    ANSWER_TOO_SHORT = "answer_too_short"
    ANSWER_TOO_LONG = "answer_too_long"
    WRONG_FORM = "wrong_form"
    WRONG_TYPE = "wrong_type"
    EXPLANATION_UNUSABLE = "explanation_unusable"
    LEAKS_SOURCE = "leaks_source"
    UNANCHORED = "unanchored"
    NOT_RECOVERABLE = "not_recoverable"
    DUPLICATE = "duplicate"
    ANSWERABLE_AFTER_ALL = "answerable_after_all"
    ANSWERABLE_ELSEWHERE = "answerable_elsewhere"
    OFF_TOPIC = "off_topic"
    ASKS_NOTHING_NEW = "asks_nothing_new"
    OFF_THREAD = "off_thread"
    SOURCE_CHANGED = "source_changed"


class FactKind(StrEnum):
    """What a fact is, and so which checks it is held to.

    ATOMIC carries one claim from one sentence. SUMMARY and OUTLINE stand in
    for a whole passage, as prose and as bullet points. BRIDGE carries one
    claim that no single passage states.
    """

    ATOMIC = "atomic"
    SUMMARY = "summary"
    OUTLINE = "outline"
    BRIDGE = "bridge"


class ReviewVerdict(StrEnum):
    """What a person decided about a fact they were shown.

    Stored in facts.reviewed_verdict, which is NULL until somebody looks.
    Separate from `validated` on purpose: that one is the checker's and
    `extract-revalidate` rewrites it in full, so a human decision recorded
    there would last until the next re-judgement and then be gone with
    nothing saying it had been. The same split questions already have
    between `status` and `rejected_reason`.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Rejection(StrEnum):
    """Why a candidate fact was not accepted.

    Stored beside the English reason so the quality report groups on a
    stable value rather than on a message carrying a measurement.
    """

    EVIDENCE_ABSENT = "evidence_absent"
    COPIED = "copied"
    ASSERTS_NOTHING = "asserts_nothing"
    NOT_ATOMIC = "not_atomic"
    UNSUPPORTED_ADDITION = "unsupported_addition"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    NOT_CONDENSED = "not_condensed"
    NOT_LISTED = "not_listed"
    NOT_BRIDGING = "not_bridging"
    OVER_CAP = "over_cap"
    DUPLICATE = "duplicate"


def one_of(column: str, values: type[StrEnum]) -> str:
    """Builds the CHECK expression constraining a column to an enum."""
    return f"{column} IN ({', '.join(f"'{value}'" for value in values)})"
