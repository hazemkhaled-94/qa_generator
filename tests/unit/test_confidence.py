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
        "high_is_safe": False,
        "safe_end": 0.0,
    }


def test_a_recorded_reading_carries_which_way_its_gate_runs() -> None:
    """0.88 is comfortable under one gate and refusing under another.

    A reader holding the number alone cannot tell, so the direction and the
    end of the scale are stored beside it rather than left to a second list
    of gates somewhere else.
    """
    (written,) = scored([Measurement("near_duplicate", 0.88, 0.93, False, 0.75)])

    assert written["high_is_safe"] is False
    assert written["safe_end"] == 0.75


def test_room_is_measured_to_where_the_scale_ends_not_to_zero() -> None:
    """A cosine's safe side stops where unrelated things score, not at 0."""
    measured = Measurement("near_duplicate", 0.88, 0.93, False, safe_end=0.75)

    # 0.05 of the 0.18 the scale actually offers, not of the 0.93 it does not.
    assert measured.margin == pytest.approx(0.05 / 0.18, abs=1e-4)


def test_a_reading_at_the_floor_has_every_bit_of_room() -> None:
    """As far from refusal as this scale goes is 1.0, wherever that sits."""
    assert Measurement("near_duplicate", 0.75, 0.93, False, safe_end=0.75).margin == 1.0


def test_a_reading_past_the_floor_is_clamped_rather_than_over_one() -> None:
    """A corpus quieter than the floor was calibrated for does not score 1.4."""
    assert Measurement("near_duplicate", 0.60, 0.93, False, safe_end=0.75).margin == 1.0


def test_a_floor_on_its_threshold_scores_zero_rather_than_raising() -> None:
    """Calibration that leaves no room divides by none of it."""
    assert Measurement("near_duplicate", 0.8, 0.93, False, safe_end=0.93).margin == 0.0


def test_the_floor_is_what_spreads_a_corpus_over_the_scale() -> None:
    """The defect this calibration exists for, at the two ends it showed up.

    Measured over 3,781 scored questions: the lowest near_duplicate reading
    was 0.745 and the median 0.882. Against a floor of 0 both land in the
    bottom fifth and are indistinguishable; against 0.75 they are a
    question worth reviewing last and one worth reviewing first.
    """
    lowest, median = 0.745, 0.882
    uncalibrated = [
        Measurement("near_duplicate", one, 0.93, False) for one in (lowest, median)
    ]
    calibrated = [
        Measurement("near_duplicate", one, 0.93, False, safe_end=0.75)
        for one in (lowest, median)
    ]

    assert all(one.margin < 0.2 for one in uncalibrated)
    assert calibrated[0].margin == 1.0
    assert calibrated[1].margin == pytest.approx(0.2667, abs=1e-4)
