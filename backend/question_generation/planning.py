"""What to write before anything is written: the plan for one topic.

A plan is a type, a difficulty band to aim for, the shape of sample that band
needs, and whether the question is meant to have an answer at all. It is
decided before the writer is called, because the prompt, the answer bounds
and the gates all differ by type.

Nothing here draws a random number. Two mixes are turned into an exact,
repeatable sequence of slots, so the same settings over the same corpus give
the same plan twice.

The band is a request, not a verdict. `difficulty` is still read off what the
question turned out to cite; `planned_difficulty` is this, stored beside it,
and the two disagreeing is a measurement of how often the plan was met.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from database.qa_generator import Difficulty
from question_generation.selection import Shape, spread
from question_generation.types import SPECS, TypeSpec

#: The sample shape each band needs, and what each is worth. A cross-document
#: question is a multi-passage question by construction, so `cross` is the two
#: points `medium` wants; a bridging passage in another document adds the
#: third subject scope, which is `hard`.
SHAPES: dict[str, str] = {
    Difficulty.EASY: Shape.SINGLE,
    Difficulty.MEDIUM: Shape.CROSS,
    Difficulty.HARD: Shape.BRIDGE,
}


@dataclass(frozen=True)
class Plan:
    """One question to write, decided before it is written."""

    spec: TypeSpec
    band: str
    shape: str
    answerable: bool

    @property
    def spans(self) -> bool:
        """Whether the writer is told to use more than one passage."""
        return self.shape != Shape.SINGLE


def allocate(weights: Mapping[str, int], slots: int) -> list[str]:
    """Spreads `slots` over named weights, proportionally and in order.

    Highest averages: each slot goes to whichever name is furthest behind its
    share so far. Over a whole run the counts are the weights; along the run
    the names are interleaved rather than run in blocks, which matters because
    other decisions - the unanswerable share, the follow-up share - are also
    taken by position and would otherwise always land on the same name.

    A weight of zero never takes a slot, which is how a type is turned off.
    """
    wanted = {name: weight for name, weight in weights.items() if weight > 0}
    if not wanted or slots <= 0:
        return []

    order = list(wanted)
    taken = dict.fromkeys(order, 0)
    chosen: list[str] = []
    for _ in range(slots):
        name = max(
            order,
            key=lambda one: (wanted[one] / (2 * taken[one] + 1), -order.index(one)),
        )
        taken[name] += 1
        chosen.append(name)
    return chosen


def plans(
    *,
    wanted: int,
    types: Mapping[str, int],
    bands: Mapping[str, int],
    unanswerable_share: float,
) -> list[Plan]:
    """The questions one topic is to be asked, in the order they are written.

    A slot that came up `easy` is raised to whatever floor its type declares.
    Usually that is `easy` and nothing moves. A type needing more than one
    passage cannot be easy by construction - the band is a request for a
    shape of sample and `easy` offers one passage, so a comparison drawn
    from one is a question about one thing. A type needing one passage may
    still decline the band: an `application` puts a rule in the material to
    a case that is not, which is not a lookup however little it reaches.

    An unanswerable question is planned `easy` and single whatever its slot
    said. It is written by moving one fact out of reach, so a second passage
    has nothing to do, and a chatbot declining it is not made cleverer by the
    question having spanned two documents.

    And its TYPE is swapped for one that does not span, because the writer is
    handed one fact to perturb: an unanswerable `comparison` has one side and
    an unanswerable `aggregation` has one number, and neither is the kind of
    question it was planned to be. The non-spanning types are cycled in mix
    order, so the swap is spread over them rather than landing on one.
    """
    typed = allocate(types, wanted)
    banded = allocate(bands, wanted)
    if not typed or not banded:
        return []

    alone = [
        name for name, weight in types.items() if weight > 0 and not SPECS[name].spans
    ]
    planned = []
    perturbed = 0
    for index, name in enumerate(typed):
        answerable = not spread(index, unanswerable_share)
        band = banded[index % len(banded)]
        if not answerable and alone:
            name = alone[perturbed % len(alone)]
            perturbed += 1
        spec = SPECS[name]
        if band == Difficulty.EASY:
            band = spec.floor
        if not answerable:
            band = Difficulty.EASY
        shape = Shape.SINGLE if not answerable else SHAPES[band]
        planned.append(Plan(spec=spec, band=band, shape=shape, answerable=answerable))
    return planned
