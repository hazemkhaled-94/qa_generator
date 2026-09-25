"""The margin arithmetic, which every stored confidence is derived from."""

from __future__ import annotations

import pytest

from confidence import Measurement, confidence, scored


def test_a_reading_far_on_the_safe_side_has_all_its_room() -> None:
    """A question sharing nothing with its nearest twin is as safe as one gets."""
    assert Measurement("near_duplicate", 0.0, 0.93, high_is_safe=False).margin == 1.0


def test_a_reading_on_its_threshold_has_none() -> None:
    """The verdict changes here, so there is no room left on the safe side."""
    assert Measurement("near_duplicate", 0.93, 0.93, high_is_safe=False).margin == 0.0


def test_a_reading_past_its_threshold_is_clamped_rather_than_negative() -> None:
    """A refused artefact has no confidence, not a negative one."""
    assert Measurement("near_duplicate", 0.99, 0.93, high_is_safe=False).margin == 0.0


def test_the_direction_is_read_off_the_gate_and_not_the_number() -> None:
    """The same value is safe for one gate and refusing for another.

    `near_duplicate` refuses a high cosine; the entailment rescue needs a
    high probability. A margin that assumed one direction would invert every
    verdict on half the gates.
    """
    refusing = Measurement("near_duplicate", 0.8, 0.7, high_is_safe=False)
    rescuing = Measurement("entailment", 0.8, 0.7, high_is_safe=True)

    assert refusing.margin == 0.0
    assert rescuing.margin == pytest.approx(1 / 3)


def test_a_threshold_leaving_no_room_scores_zero_rather_than_raising() -> None:
    """An absolute threshold has no safe side to be a share of.

    Nothing may be above 1.0 and nothing below 0.0, so the reading that
    needs one has no room whichever way it went. It scores zero rather than
    dividing by it. A gate a deployment turned OFF - which in this codebase
    is a threshold of 0 - should record no reading at all; that is the
    caller's decision and not this arithmetic's.
    """
    assert Measurement("entailment", 0.5, 1.0, high_is_safe=True).margin == 0.0
    assert Measurement("near_duplicate", 0.5, 0.0, high_is_safe=False).margin == 0.0


def test_the_aggregate_is_the_weakest_reading() -> None:
    """One number can only honestly say what the worst thing about a row is."""
    readings = [
        Measurement("near_duplicate", 0.10, 0.93, high_is_safe=False),
        Measurement("about", 0.35, 0.30, high_is_safe=True),
    ]

    assert confidence(readings) == pytest.approx(
        Measurement("about", 0.35, 0.30).margin, abs=1e-4
    )


def test_nothing_measured_is_not_the_same_as_measured_badly() -> None:
    """None keeps an unmeasured row out of the bottom of a review queue."""
    assert confidence([]) is None


def test_a_recorded_reading_carries_its_threshold() -> None:
    """A value alone cannot be read: 0.71 is safe under one gate and not another."""
    (written,) = scored([Measurement("near_duplicate", 0.71, 0.93, high_is_safe=False)])

    assert written == {
        "gate": "near_duplicate",
        "value": 0.71,
        "threshold": 0.93,
        "margin": pytest.approx(0.2366, abs=1e-4),
    }
