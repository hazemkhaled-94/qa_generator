"""How close an artefact came to the verdict that would have refused it.

Every gate in this pipeline that is a MEASUREMENT reads a number and
compares it to a threshold, and until now only the comparison survived. A
question 0.92 from an accepted twin and one 0.01 from it are both stored as
`accepted` against a 0.93 ceiling, and nothing anywhere says which of the
two a reviewer should look at first.

**This is not a probability and must not be read as one.** The readings are
on different scales - a cosine, an NLI head's entailment output, a share of
lemmas - and averaging them would invent a number none of them supports.
What is derived instead is the MARGIN: how far one reading sat from its own
threshold, as a share of the room it had on the safe side. That is a
comparable quantity because it is dimensionless, and the aggregate is the
SMALLEST of them, which answers the only question a single number honestly
can - what is the weakest thing about this row.

A gate that is a judgement rather than a measurement contributes nothing
here, and that is the point. `leaks_source` answered by a model is a yes or
a no with no number behind it; recording 1.0 for it would put a confidence
on an opinion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Measurement:
    """One gate's measurement, against the threshold it was read for.

    Attributes:
        gate: The gate that took it, named as `gates_ran` names it.
        value: What was measured, on that gate's own scale.
        threshold: The value at which the verdict changes.
        high_is_safe: Whether a HIGH reading is what keeps the artefact.
            The entailment rescue needs one; `near_duplicate` refuses one.
    """

    gate: str
    value: float
    threshold: float
    high_is_safe: bool = True

    @property
    def margin(self) -> float:
        """How much room this reading had, as a share of the room available.

        1.0 is as far from refusal as the scale allows, 0.0 is on the line
        or past it. Clamped at both ends: a rejected artefact has no
        negative confidence, it has none at all, and a threshold of 0 or 1
        leaves no room to be a share of and yields 0.0 rather than a
        division by zero.
        """
        room = 1.0 - self.threshold if self.high_is_safe else self.threshold
        if room <= 0:
            return 0.0
        distance = (
            self.value - self.threshold
            if self.high_is_safe
            else self.threshold - self.value
        )
        return max(0.0, min(1.0, distance / room))

    def recorded(self) -> dict[str, Any]:
        """This reading as the JSON column stores it.

        `margin` is written out rather than left to be re-derived. Every
        reader of the column would otherwise need this module's arithmetic,
        including the ones that are SQL and the ones that are a dashboard.
        """
        return {
            "gate": self.gate,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "margin": round(self.margin, 4),
        }


def scored(readings: Sequence[Measurement]) -> list[dict[str, Any]]:
    """Every reading as the column stores them, in the order they were taken."""
    return [one.recorded() for one in readings]


def confidence(readings: Sequence[Measurement]) -> float | None:
    """The weakest margin of the lot, or None when nothing was measured.

    None rather than 0.0, and the distinction is load-bearing: a question
    no measuring gate read is not a question that barely survived one. It
    is a question whose gates were all judgements or all free rules, and a
    column of zeroes would sort those to the top of every review queue.
    """
    if not readings:
        return None
    return round(min(one.margin for one in readings), 4)
