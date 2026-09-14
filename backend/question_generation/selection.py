"""Turning one topic's facts into the groups a question is written from.

Two decisions, both deterministic: which facts go together, and which of the
resulting questions is asked of the corpus rather than answered by it. Neither
draws a random number. The same corpus and the same settings give the same
questions twice, for the reason TOPIC_RANDOM_STATE exists - a reference
dataset whose contents move between runs is not a reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest

from question_generation.models import FactGroup, SourceFact


def groups(facts: Iterable[SourceFact], *, wanted: int, size: int) -> list[FactGroup]:
    """Forms up to `wanted` groups of at most `size` facts.

    Facts are dealt one document at a time in turn rather than taken in
    order. Reading straight down the list would fill every group from one
    document and the corpus would never be crossed, which is the whole point
    of a group larger than one: a question needing two documents is a
    question a retriever has to actually retrieve for.

    A corpus holding one document still yields groups - of two facts from
    that document, which is a cross-passage question when they come from
    different passages.

    A size below one is read as one rather than refused. Nothing downstream
    can do anything with an empty group - it has no language to write in and
    no spread to read a difficulty off - so the one thing this must not do
    is hand one out.
    """
    held = max(size, 1)
    dealt = _interleaved(facts)
    return [
        FactGroup(tuple(dealt[at : at + held])) for at in range(0, len(dealt), held)
    ][:wanted]


def _interleaved(facts: Iterable[SourceFact]) -> list[SourceFact]:
    """Every fact, taken one document at a time in turn.

    Sorted throughout: by document and then by fact id, so the deal is the
    same deal on the same rows whatever order the database returned them in.
    """
    by_document: dict[str, list[SourceFact]] = {}
    for fact in sorted(facts, key=lambda one: (one.doc_sha256, one.id)):
        by_document.setdefault(fact.doc_sha256, []).append(fact)
    return [
        fact
        for row in zip_longest(*by_document.values())
        for fact in row
        if fact is not None
    ]


def perturbed(index: int, share: float) -> bool:
    """Whether the group at this position is perturbed rather than asked.

    Spread across the run by position instead of drawn at random, so a share
    of 0.25 gives exactly one unanswerable question in every four and gives
    the same four twice. 0 never perturbs and 1 always does, without either
    needing a branch of its own.
    """
    return int((index + 1) * share) > int(index * share)
