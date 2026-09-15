"""Turning one topic's facts into the samples a question is written from.

Two decisions, both deterministic: which facts are offered together, and
which of the resulting questions is asked of the corpus rather than answered
by it. Neither draws a random number. The same corpus and the same settings
give the same questions twice, for the reason TOPIC_RANDOM_STATE exists - a
reference dataset whose contents move between runs is not a reference.

What this does not decide is which facts a question is *written from*. A
sample is an offer of several related facts; the writer says which of them it
used, and the difficulty is read off those. Deciding here instead was wrong
in a way worth recording: facts dealt one document at a time in turn are
spread across the corpus but are not about the same thing, so pairing them
produced a question about the first fact carrying a `cross_document` label
earned by the second.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest

from question_generation.models import FactGroup, SourceFact


def bridged(
    formed: list[FactGroup],
    bridges: Iterable[SourceFact],
    *,
    share: float,
    size: int,
) -> list[FactGroup]:
    """Adds a bridging passage to a share of the samples.

    A bridge is a passage whose strongest topic is not the one being worked,
    but which carries it above the weight floor. Adding one to a sample is
    what makes a multi-topic question available: the two passages are about
    different subjects and the corpus itself says they are related, which is
    a far better reason to pair them than that they came from different
    files.

    Only a share, by position, and only where the sample has room. A bridge
    on every sample would make `single_topic` unreachable, and the scope is
    worth measuring in both directions.
    """
    available = _by_passage(bridges)
    if not available:
        return formed

    widened: list[FactGroup] = []
    taken = 0
    for index, sample in enumerate(formed):
        if not spread(index, share) or taken >= len(available):
            widened.append(sample)
            continue
        passage = available[taken]
        taken += 1
        # Room for the whole passage or none of it, as when a sample is
        # filled: half a passage's facts is the incoherence to avoid.
        if len(sample.facts) + len(passage) > max(size, 1):
            widened.append(sample)
            continue
        widened.append(FactGroup((*sample.facts, *passage)))
    return widened


def samples(facts: Iterable[SourceFact], *, wanted: int, size: int) -> list[FactGroup]:
    """Forms up to `wanted` samples of at most `size` facts to offer.

    A sample is filled a whole passage at a time, not a fact at a time. That
    is what gives the writer something to write a meaningful question from:
    every fact of a passage is about the same material, so the writer sees a
    subject rather than a list of unrelated claims and can ask about the
    subject. Filling fact by fact across documents - which this did first -
    offered claims that had nothing to do with each other, and the writer
    correctly answered one and ignored the rest.

    Passages are taken one document at a time in turn, so a sample spans the
    corpus wherever the topic does and a cross-document question is there to
    be written. Whether one is written is the writer's: it is handed the
    sample and says which facts a single question needs.

    `size` caps the facts offered, not the passages. A passage is added whole
    or not at all, because half a passage's facts is the incoherence this
    exists to avoid; one passage carrying more facts than the cap is offered
    alone rather than dropped.
    """
    held = max(size, 1)
    formed: list[FactGroup] = []
    current: list[SourceFact] = []

    for passage in _by_passage(facts):
        if current and len(current) + len(passage) > held:
            formed.append(FactGroup(tuple(current)))
            current = []
            if len(formed) >= wanted:
                return formed
        current.extend(passage)

    if current:
        formed.append(FactGroup(tuple(current)))
    return formed[:wanted]


def _by_passage(facts: Iterable[SourceFact]) -> list[list[SourceFact]]:
    """Each passage's facts, the passages taken one document at a time in turn.

    Sorted throughout - by document, then passage, then fact id - so the deal
    is the same deal on the same rows whatever order the database returned
    them in.
    """
    passages: dict[tuple[str, int], list[SourceFact]] = {}
    for fact in sorted(facts, key=lambda one: (one.doc_sha256, one.passage_id, one.id)):
        passages.setdefault((fact.doc_sha256, fact.passage_id), []).append(fact)

    by_document: dict[str, list[list[SourceFact]]] = {}
    for (document, _), held in passages.items():
        by_document.setdefault(document, []).append(held)
    return [
        held
        for row in zip_longest(*by_document.values())
        for held in row
        if held is not None
    ]


def spread(index: int, share: float) -> bool:
    """Whether the thing at this position is one of a share of them.

    Spread across the run by position instead of drawn at random, so a share
    of 0.25 picks exactly one in every four and picks the same four twice. 0
    never picks and 1 always does, without either needing a branch of its
    own.

    Three decisions read it: which samples are perturbed into unanswerable
    questions, which get a bridging passage, and which accepted questions
    get a follow-up. All three want an exact, reproducible share of a run,
    which is the same want.
    """
    return int((index + 1) * share) > int(index * share)
