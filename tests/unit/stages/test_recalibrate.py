"""Rebuilding a stored reading.

The half of a recalibration that is not a database write.
"""

from __future__ import annotations

import pytest

from stages.recalibrate import _reading


def test_a_legacy_reading_gets_its_direction_from_the_gate_name() -> None:
    """Rows written before the direction was stored carry only four keys."""
    rebuilt = _reading(
        {"gate": "near_duplicate", "value": 0.88, "threshold": 0.93, "margin": 0.0538},
        floor=0.75,
    )

    assert rebuilt.high_is_safe is False
    assert rebuilt.safe_end == 0.75
    assert rebuilt.margin == pytest.approx(0.05 / 0.18, abs=1e-4)


def test_a_legacy_reading_of_a_gate_read_the_other_way_keeps_its_end() -> None:
    """The floor is a similarity scale's, and `recall` is not one.

    Applying it to every gate would move a reading no calibration was
    wrong about, which is the way a recalibration does damage.
    """
    rebuilt = _reading(
        {"gate": "recall", "value": 0.8, "threshold": 0.6, "margin": 0.5}, floor=0.75
    )

    assert rebuilt.high_is_safe is True
    assert rebuilt.safe_end is None
    assert rebuilt.margin == pytest.approx(0.5, abs=1e-4)


def test_a_recorded_direction_is_believed_over_the_name() -> None:
    """A row written since carries its own direction, so the map is not read."""
    rebuilt = _reading(
        {
            "gate": "invented_later",
            "value": 0.8,
            "threshold": 0.93,
            "margin": 0.0,
            "high_is_safe": False,
            "safe_end": 0.75,
        },
        floor=0.75,
    )

    assert rebuilt.high_is_safe is False
    assert rebuilt.safe_end == 0.75


def test_an_unknown_legacy_gate_is_read_as_the_common_direction() -> None:
    """High is safe for every gate here but the two similarity ones."""
    rebuilt = _reading(
        {"gate": "something_new", "value": 0.8, "threshold": 0.6, "margin": 0.5},
        floor=0.75,
    )

    assert rebuilt.high_is_safe is True


def test_the_command_reads_both_floors_from_the_settings(monkeypatch) -> None:
    """The entry point's whole job: two settings, one call, a reported total.

    Stubbed at `main`, because what the command contributes is reading the
    right two settings and reporting what came back - the write itself is
    `tests/integration/database/test_recalibration_store.py`'s.
    """
    from stages import recalibrate_run

    monkeypatch.setenv("QUESTIONS_DUPLICATE_FLOOR", "0.75")
    monkeypatch.setenv("EXTRACTION_DUPLICATE_FLOOR", "0.70")
    asked: dict[str, float] = {}

    def recorded(questions_floor: float, facts_floor: float) -> int:
        """Stands in for the recalibration, keeping what it was asked."""
        asked.update(questions=questions_floor, facts=facts_floor)
        return 7

    monkeypatch.setattr(recalibrate_run, "main", recorded)

    assert recalibrate_run.run() == 0
    assert asked == {"questions": 0.75, "facts": 0.70}
