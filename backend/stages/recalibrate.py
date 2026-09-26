"""Recomputing a stored confidence when its calibration changes.

`gate_scores` records what each measuring gate read and what it was read
against, so a margin is arithmetic over two numbers already in the row.
Nothing here calls a model, an embedder or a service: a recalibration is a
read and a write over one column.

It is separate from every stage's own re-check for that reason.
`question_generation.reverify` puts a stored question through the gates
that need no model and may REJECT it; this moves no verdict and touches
neither `status` nor `rejected_reason`.

A row written before the direction was recorded carries only `gate`,
`value`, `threshold` and `margin`, so which way its gate runs has to come
from somewhere. `_REFUSES_HIGH` is that somewhere, and it is the only place
in the codebase that has to know: every row written since carries its own
`high_is_safe`.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from confidence import Measurement, confidence, scored
from database.qa_generator import Fact, Question, sessions

log = logging.getLogger(__name__)

#: The gates a HIGH reading refuses, for rows written before the direction
#: was stored beside the reading. Everything else keeps what it measured by
#: reading high.
_REFUSES_HIGH = frozenset({"near_duplicate", "duplicate"})


def _reading(stored: dict, floor: float) -> Measurement:
    """One stored reading, rebuilt so its margin can be taken again.

    The direction comes off the row where the row carries one, and off
    `_REFUSES_HIGH` where it does not. `floor` applies only to a gate a
    high reading refuses: it is where a similarity scale ends, and the
    gates read the other way already end at 1.0.
    """
    refuses_high = stored.get("high_is_safe") is False or (
        "high_is_safe" not in stored and stored["gate"] in _REFUSES_HIGH
    )
    return Measurement(
        gate=stored["gate"],
        value=float(stored["value"]),
        threshold=float(stored["threshold"]),
        high_is_safe=not refuses_high,
        safe_end=floor if refuses_high else None,
    )


def recalibrate(model, floor: float) -> int:
    """Takes every stored row's margins again, and returns how many moved.

    Args:
        model: `Question` or `Fact` - the two tables carrying the columns.
        floor: Where the similarity scale ends for the gate that refuses a
            high reading.

    Returns:
        How many rows were written. A row whose readings come back
        unchanged is left alone, so running this twice writes nothing the
        second time.
    """
    written = 0
    with sessions().begin() as session:
        rows = session.scalars(
            select(model).where(model.gate_scores.is_not(None))
        ).all()
        for row in rows:
            stored = list(row.gate_scores or ())
            if not stored:
                continue
            readings = [_reading(one, floor) for one in stored]
            recomputed = scored(readings)
            if recomputed == stored:
                continue
            row.gate_scores = recomputed
            row.confidence = confidence(readings)
            written += 1
    log.info("recalibrated %d %s row(s)", written, model.__tablename__)
    return written


def main(questions_floor: float, facts_floor: float) -> int:
    """Recalibrates both tables, reporting what each wrote."""
    return recalibrate(Question, questions_floor) + recalibrate(Fact, facts_floor)
