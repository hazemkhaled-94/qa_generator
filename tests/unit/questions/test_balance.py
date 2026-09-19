"""Choosing a balanced release out of everything a run accepted.

The gates decide whether a question is sound; nothing before this decides
what the SET looks like. Over one corpus every gate did its job and the
accepted set still came out 45% unanswerable and 96.5% easy, because what
survives a filter is whatever the material happened to offer.

Three marginals are held at once - answerability, difficulty band and kind -
and they are marginals rather than a joint distribution on purpose. An even
spread of kinds WITHIN each band is not reachable: a factoid answers with a
value, a value is short, and a short answer cannot earn the length point a
hard question generally needs.
"""

from __future__ import annotations

import pytest

from database.qa_generator import Difficulty, QuestionType
from question_generation.balance import (
    Release,
    Row,
    choose,
    close_enough,
    composition,
    largest,
    quota_of,
    shares,
)

BANDS = {Difficulty.EASY: 1, Difficulty.MEDIUM: 1, Difficulty.HARD: 1}
TYPES = {QuestionType.FACTOID: 1, QuestionType.REASON: 1}


def row(
    id: int,
    band: str = Difficulty.EASY,
    kind: str = QuestionType.FACTOID,
    answerable: bool = True,
) -> Row:
    """One accepted question, as the choosing reads it."""
    return Row(id=id, answerable=answerable, difficulty=band, question_type=kind)


def pool(counts: dict[tuple[str, str, bool], int]) -> list[Row]:
    """A pool holding this many questions of each band, kind and answerability."""
    built: list[Row] = []
    for (band, kind, answerable), count in counts.items():
        for _ in range(count):
            built.append(row(len(built) + 1, band, kind, answerable))
    return built


# ── Spreading a size over weights ──────────────────────────────────────────


def test_whole_shares_are_exact() -> None:
    """Nothing to round, so nothing is rounded."""
    assert shares({"a": 1, "b": 1}, 10) == {"a": 5, "b": 5}


def test_the_shares_always_sum_to_the_size() -> None:
    """A quota that does not add up is a release short of its own target."""
    for size in range(1, 60):
        assert sum(shares(BANDS, size).values()) == size


def test_the_remainder_goes_to_the_largest_fractions() -> None:
    """Largest remainder, so the spread is as even as the size allows."""
    got = shares({"a": 1, "b": 1, "c": 1}, 10)

    assert sum(got.values()) == 10
    assert sorted(got.values()) == [3, 3, 4]


def test_the_same_weights_give_the_same_answer_twice() -> None:
    """Ties by name, so a release is reproducible from its settings."""
    assert shares({"a": 1, "b": 1, "c": 1}, 10) == shares({"c": 1, "b": 1, "a": 1}, 10)


@pytest.mark.parametrize("size", [0, -1])
def test_no_size_is_no_quota(size) -> None:
    """Rather than a negative one, or a division by nothing."""
    assert set(shares(BANDS, size).values()) == {0}


def test_a_weight_of_zero_takes_no_place() -> None:
    """Which is how a deployment turns a type off."""
    assert shares({"a": 1, "b": 0}, 10) == {"a": 10, "b": 0}


# ── Filling a quota ────────────────────────────────────────────────────────


def test_a_pool_that_matches_the_quota_fills_it_exactly() -> None:
    """The case every shortfall below is measured against."""
    counted = {
        (band, kind, True): 10
        for band in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        for kind in (QuestionType.FACTOID, QuestionType.REASON)
    }

    release = choose(pool(counted), quota_of(6, BANDS, TYPES, 0.0))

    assert release.size == 6
    assert release.short == {}


def test_the_unanswerable_share_is_a_ceiling() -> None:
    """Too many and the release mostly tests whether a chatbot can say no."""
    counted = {
        (Difficulty.EASY, QuestionType.FACTOID, False): 50,
        (Difficulty.EASY, QuestionType.FACTOID, True): 50,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 50,
        (Difficulty.HARD, QuestionType.FACTOID, True): 50,
        (Difficulty.EASY, QuestionType.REASON, True): 50,
        (Difficulty.MEDIUM, QuestionType.REASON, True): 50,
        (Difficulty.HARD, QuestionType.REASON, True): 50,
    }
    held = pool(counted)

    release = choose(held, quota_of(100, BANDS, TYPES, 0.10))
    got = composition(held, release.ids)

    assert release.size == 100
    assert got["answerable"]["unanswerable"] == 10


def test_a_band_the_pool_cannot_supply_is_reported_short() -> None:
    """Rather than a release quietly missing a third of what it claims."""
    counted = {
        (Difficulty.EASY, QuestionType.FACTOID, True): 50,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 50,
        (Difficulty.EASY, QuestionType.REASON, True): 50,
        (Difficulty.MEDIUM, QuestionType.REASON, True): 50,
    }

    release = choose(pool(counted), quota_of(30, BANDS, TYPES, 0.0))

    assert release.short.get(Difficulty.HARD) == 10
    assert release.size == 20


def test_a_scarce_pair_gets_its_place_before_a_plentiful_one() -> None:
    """The reason the pool is walked scarcest first.

    Taken in id order the common pairs spend the whole `reason` quota on easy
    ones, and the single hard reason - the only question that can fill that
    cell - is passed over because its type is full.
    """
    counted = {
        (Difficulty.EASY, QuestionType.REASON, True): 40,
        (Difficulty.HARD, QuestionType.REASON, True): 1,
        (Difficulty.EASY, QuestionType.FACTOID, True): 40,
        (Difficulty.HARD, QuestionType.FACTOID, True): 40,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 40,
        (Difficulty.MEDIUM, QuestionType.REASON, True): 40,
    }
    held = pool(counted)

    release = choose(held, quota_of(6, BANDS, TYPES, 0.0))
    taken = {one.id for one in held if one.id in set(release.ids)}
    hard_reasons = [
        one
        for one in held
        if one.id in taken
        and one.difficulty == Difficulty.HARD
        and one.question_type == QuestionType.REASON
    ]

    assert release.short == {}
    assert len(hard_reasons) == 1


# ── The largest release a pool can fill ────────────────────────────────────


def test_the_largest_release_comes_within_a_place_of_every_quota() -> None:
    """Which is what makes it the largest one worth shipping.

    One place per bucket, not zero: a quota of eight cannot absorb a
    proportional tolerance, and being one short of eight is where the
    rounding fell rather than a pool that cannot supply the kind.
    """
    counted = {
        (Difficulty.EASY, QuestionType.FACTOID, True): 30,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 30,
        (Difficulty.HARD, QuestionType.FACTOID, True): 5,
        (Difficulty.EASY, QuestionType.REASON, True): 30,
        (Difficulty.MEDIUM, QuestionType.REASON, True): 30,
        (Difficulty.HARD, QuestionType.REASON, True): 5,
    }
    held = pool(counted)

    release = largest(held, BANDS, TYPES, 0.0)
    got = composition(held, release.ids)

    assert close_enough(release)
    assert all(short <= 1 for short in release.short.values())
    # Ten hard questions is what caps it, give or take the one place each
    # bucket may be short.
    assert 30 <= release.size <= 34
    assert got["difficulty"][Difficulty.HARD] == 10


def test_the_scarcest_band_is_what_caps_the_size() -> None:
    """A pool that is 12% hard fills no release that wants a third hard."""
    counted = {
        (Difficulty.EASY, QuestionType.FACTOID, True): 100,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 100,
        (Difficulty.HARD, QuestionType.FACTOID, True): 3,
    }

    release = largest(pool(counted), BANDS, {QuestionType.FACTOID: 1}, 0.0)

    # Three hard questions, and one place of slack per bucket on top.
    assert close_enough(release)
    assert 9 <= release.size <= 13


def test_an_empty_pool_releases_nothing() -> None:
    """Rather than bisecting over nothing and returning a quota it cannot fill."""
    release = largest([], BANDS, TYPES, 0.0)

    assert release.ids == []
    assert release.size == 0


def test_a_release_may_come_a_little_short_of_its_quota() -> None:
    """Three marginals over eleven kinds cannot all reach zero together.

    Held to exactness the bisection collapsed: a pool filling 98% of a
    release of 900 was refused, and the largest it would take was 199. The
    target is "about a third each", so a few places short is taken and
    reported rather than thrown away.
    """
    counted = {
        (band, kind, True): 60
        for band in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        for kind in (QuestionType.FACTOID, QuestionType.REASON)
    }
    held = pool(counted)

    release = largest(held, BANDS, TYPES, 0.0)
    got = composition(held, release.ids)
    bands = got["difficulty"]

    # Nearly the whole pool, and every band within a few points of a third.
    assert release.size > len(held) * 0.9
    assert max(bands.values()) - min(bands.values()) <= release.size * 0.05


def test_what_it_could_not_fill_is_still_reported() -> None:
    """The tolerance is not a reason to stop saying what was missing."""
    counted = {
        (Difficulty.EASY, QuestionType.FACTOID, True): 50,
        (Difficulty.MEDIUM, QuestionType.FACTOID, True): 50,
        (Difficulty.EASY, QuestionType.REASON, True): 50,
        (Difficulty.MEDIUM, QuestionType.REASON, True): 50,
    }

    release = choose(pool(counted), quota_of(30, BANDS, TYPES, 0.0))

    assert release.short.get(Difficulty.HARD) == 10


def test_the_tolerance_is_set_above_the_endgame_not_below_it() -> None:
    """A tolerance under the shortfall buys almost no release at all.

    The last places of a draw cannot be filled - three marginals over
    eleven kinds do not reach zero together - and that endgame is about 5%
    of each band at every size. Held to 3%, a pool of 2,259 questions gave
    a largest release of 58; at 8% the same questions gave 1,145, with the
    bands closer to even. So this is a floor as much as a ceiling.
    """
    counted = {
        (band, kind, True): 60
        for band in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        for kind in (QuestionType.FACTOID, QuestionType.REASON)
    }
    held = pool(counted)

    release = largest(held, BANDS, TYPES, 0.0)

    assert release.size > len(held) * 0.9


def test_a_bucket_the_pool_genuinely_cannot_supply_is_still_refused() -> None:
    """Loosening the tolerance must not turn the check off.

    The draw it exists to refuse was six aggregations short of fourteen -
    43% of that bucket - which is what an unbalanced set looks like beside
    an endgame of 5%.
    """
    short = Release(
        ids=[1],
        quota=quota_of(140, BANDS, {**TYPES, QuestionType.AGGREGATION: 1}, 0.0),
        short={QuestionType.AGGREGATION: 6},
    )

    assert not close_enough(short)
