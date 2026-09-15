"""Turning one topic's facts into the samples a question is written from.

Deterministic throughout: nothing draws a random number, so the same corpus
and the same settings give the same samples twice. A reference dataset whose
contents move between runs is not a reference.

A sample is an OFFER. Which facts a question is written from is the writer's
answer, not this module's: it is handed a sample and reports which facts one
question needed, and the difficulty is read off those.

Three things decide what is offered. The plan asks for a shape - one passage,
a neighbouring one, or one in another document. `size` caps the facts, and it
caps them per passage rather than by dropping a passage whole: the cap used to
flush a group and then add the next passage entire, so a corpus whose median
passage carries ten facts never produced a sample of two passages at all.
Passages are dealt strided over the whole topic rather than from its start, so
a topic of eighty passages is asked about across all of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import zip_longest

from question_generation.models import FactGroup, SourceFact


class Shape(StrEnum):
    """How wide a sample is.

    Each is one more scope above one, and so one more point of difficulty,
    which is what makes a band reachable at all: a question drawn from one
    passage cannot be cross-document however it is phrased.

      single  one passage                                       0 points
      cross   a passage in another document                     2 points
      bridge  one in another document AND another subject       3 points

    `cross` is two points and not one because a cross-document question is a
    multi-passage question by construction. A topic that cannot supply a
    shape gets the widest one it can: a topic sitting in one document has no
    cross-document question in it, and writing nothing about that subject is
    the worse answer.
    """

    SINGLE = "single"
    CROSS = "cross"
    BRIDGE = "bridge"


def spread(index: int, share: float) -> bool:
    """Whether the thing at this position is one of a share of them.

    Spread across the run by position instead of drawn at random, so a share
    of 0.25 picks exactly one in every four and picks the same four twice. 0
    never picks and 1 always does, without either needing a branch of its own.
    """
    return int((index + 1) * share) > int(index * share)


def overlap(left: Sequence[SourceFact], right: Sequence[SourceFact]) -> float:
    """How much vocabulary two passages share, in [0, 1].

    Jaccard over the content lemmas chunking stored, which is the same
    vocabulary the topics were fitted over. It is what makes a pair of
    passages related rather than merely distant: pairing two passages because
    they came from different files produced questions about the first that
    carried a spread earned by the second.
    """
    first, second = set(left[0].lemmas), set(right[0].lemmas)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def ranked(facts: Iterable[SourceFact]) -> list[SourceFact]:
    """One passage's facts, the ones asserting a value first.

    A fact carrying a number, a date or an amount is what a checkable question
    is written from, so it is offered before one that carries none.
    """
    return sorted(facts, key=lambda one: (not one.units, one.id))


def by_passage(facts: Iterable[SourceFact]) -> list[list[SourceFact]]:
    """Each passage's facts, ranked, the passages in corpus order."""
    held: dict[int, list[SourceFact]] = {}
    ordered = sorted(
        facts, key=lambda one: (one.doc_sha256, one.ordinal, one.passage_id, one.id)
    )
    for fact in ordered:
        held.setdefault(fact.passage_id, []).append(fact)
    return [ranked(group) for group in held.values()]


def interleaved(passages: list[list[SourceFact]]) -> list[list[SourceFact]]:
    """The passages, taken one document at a time in turn.

    So a run over a topic reaches every document the topic sits in before it
    reaches any document twice.
    """
    by_document: dict[str, list[list[SourceFact]]] = {}
    for passage in passages:
        by_document.setdefault(passage[0].doc_sha256, []).append(passage)
    return [
        passage
        for row in zip_longest(*by_document.values())
        for passage in row
        if passage is not None
    ]


def strided(items: list, wanted: int) -> list:
    """The items reordered so the first `wanted` are spread over all of them.

    Nothing is dropped: the rest follow in their own order, so asking for more
    than were spread still reaches them.
    """
    if wanted <= 0 or wanted >= len(items):
        return items
    step = len(items) / wanted
    picked = sorted({int(position * step) for position in range(wanted)})
    rest = [index for index in range(len(items)) if index not in set(picked)]
    return [items[index] for index in picked + rest]


class Deal:
    """One topic's passages, handed out as samples without repeating one.

    A passage is offered once per run. Two questions written from one passage
    are two questions about the same few sentences, and the dedup gate pays
    for both before throwing one away.
    """

    def __init__(
        self,
        facts: Iterable[SourceFact],
        bridges: Iterable[SourceFact] = (),
        *,
        wanted: int,
        size: int,
    ) -> None:
        """Deals one topic's facts, strided over the whole of it."""
        self._order = strided(interleaved(by_passage(facts)), wanted)
        self._bridges = interleaved(by_passage(bridges))
        self._size = max(size, 1)
        self._used: set[int] = set()

    @property
    def passages(self) -> int:
        """How many passages are left to offer."""
        return sum(1 for passage in self._order if not self._taken(passage))

    def sample(self, shape: str = Shape.SINGLE) -> FactGroup | None:
        """The next sample of the shape asked for, or None when none is left.

        A shape the topic cannot supply falls back to a narrower one rather
        than yielding nothing: a topic sitting in one document has no
        cross-document question in it, and refusing to write anything about it
        would leave the subject uncovered.
        """
        head = self._next()
        if head is None:
            return None
        if shape == Shape.SINGLE:
            return self._group([head])

        partner = self._bridge(head) if shape == Shape.BRIDGE else self._cross(head)
        return self._group([head, partner] if partner else [head])

    def _taken(self, passage: list[SourceFact]) -> bool:
        """Whether this passage has already been offered."""
        return passage[0].passage_id in self._used

    def _next(self) -> list[SourceFact] | None:
        """The next passage nothing has been written from."""
        for passage in self._order:
            if not self._taken(passage):
                return passage
        return None

    def _cross(self, head: list[SourceFact]) -> list[SourceFact] | None:
        """A passage in another document, or the nearest one in this document.

        Ranked by shared vocabulary, so the pair is two passages the corpus
        itself says are about related things. Pairing two passages because
        they came from different files produced questions about the first
        carrying a spread earned by the second.

        The fallback is the closest passage of the same document, by ordinal:
        passages are numbered in reading order, so a neighbour is the rest of
        the same section rather than an unrelated part of the same file. It is
        one point rather than two, and it is what a topic sitting in one
        document has.
        """
        return self._best(head, self._free(head, elsewhere=True)) or self._neighbour(
            head
        )

    def _bridge(self, head: list[SourceFact]) -> list[SourceFact] | None:
        """A bridging passage, preferably in another document.

        A bridge belongs to some other topic more strongly than to this one
        but carries this one above the weight floor, so the corpus itself says
        the two subjects meet there. One in another document is in another
        document AND another subject, which is every scope above one at once.
        """
        return (
            self._best(head, self._free(head, elsewhere=True, bridging=True))
            or self._best(head, self._free(head, bridging=True))
            or self._cross(head)
        )

    def _free(
        self,
        head: list[SourceFact],
        *,
        elsewhere: bool = False,
        bridging: bool = False,
    ) -> list[list[SourceFact]]:
        """The passages still to be had, narrowed to what a shape wants."""
        return [
            passage
            for passage in (self._bridges if bridging else self._order)
            if not self._taken(passage)
            and passage[0].passage_id != head[0].passage_id
            and (not elsewhere or passage[0].doc_sha256 != head[0].doc_sha256)
        ]

    @staticmethod
    def _best(
        head: list[SourceFact], candidates: list[list[SourceFact]]
    ) -> list[SourceFact] | None:
        """Whichever candidate shares most vocabulary with the head, or None.

        Ties go to the lower passage id, so the same corpus deals the same
        pair twice.
        """
        if not candidates:
            return None
        return max(candidates, key=lambda one: (overlap(head, one), -one[0].passage_id))

    def _neighbour(self, head: list[SourceFact]) -> list[SourceFact] | None:
        """The closest unused passage of the same document, by ordinal."""
        candidates = [
            passage
            for passage in self._free(head)
            if passage[0].doc_sha256 == head[0].doc_sha256
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda one: (
                abs(one[0].ordinal - head[0].ordinal),
                one[0].passage_id,
            ),
        )

    def _group(self, passages: list[list[SourceFact]]) -> FactGroup:
        """Marks these passages used and offers a capped share of each.

        The cap is divided between the passages rather than applied to the
        sample, so a wide sample offers both sides of what it is asking about
        instead of filling itself from the first passage.
        """
        each = max(1, self._size // len(passages))
        facts: list[SourceFact] = []
        for passage in passages:
            self._used.add(passage[0].passage_id)
            facts.extend(passage[:each])
        return FactGroup(tuple(facts))
