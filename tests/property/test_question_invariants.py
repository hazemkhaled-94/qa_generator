"""Invariants of the arithmetic question generation is built on.

The example tests say what these functions do to the corpus in front of
them. These say what must hold for any corpus at all - a quota that sums to
the size asked for, an ordering that drops nothing, a band that agrees with
its own score, and a figure reader that reads a number wherever a sentence
happens to put it.

Every one of them is pure and total, so there is nothing to stub and no
deadline worth setting.
"""

from __future__ import annotations

from math import isinf, isnan

from hypothesis import given
from hypothesis import strategies as st

from database.qa_generator import Difficulty, QuestionType
from question_generation.balance import Quota, Row, choose, quota_of, shares
from question_generation.gates import figures
from question_generation.models import band, criteria_of
from question_generation.selection import spread, strided

#: A mix as a setting writes one: names nobody has to be told, weights that
#: may include a zero, and at least one that is not.
WEIGHTS = st.dictionaries(
    st.sampled_from("abcdefg"), st.integers(0, 9), min_size=1, max_size=7
).filter(lambda mix: any(weight > 0 for weight in mix.values()))


# ── Spreading a size over named weights ──────────────────────────────────


@given(weights=WEIGHTS, size=st.integers(0, 500))
def test_a_spread_sums_to_exactly_the_size_asked_for(weights, size) -> None:
    """Largest remainder: the leftovers are handed out, not dropped."""
    assert sum(shares(weights, size).values()) == size


@given(weights=WEIGHTS, size=st.integers(0, 500))
def test_a_spread_names_every_weight_and_invents_none(weights, size) -> None:
    """A caller reads the answer by name, so a missing key is a KeyError."""
    assert set(shares(weights, size)) == set(weights)


@given(weights=WEIGHTS, size=st.integers(0, 500))
def test_nothing_is_given_a_negative_share(weights, size) -> None:
    """A share below zero is a quota that can never be filled."""
    assert all(count >= 0 for count in shares(weights, size).values())


@given(weights=WEIGHTS, size=st.integers(1, 500))
def test_a_weight_of_zero_never_takes_a_place(weights, size) -> None:
    """Which is how a type is turned off."""
    given_out = shares(weights, size)

    assert all(given_out[name] == 0 for name, w in weights.items() if w <= 0)


@given(weights=WEIGHTS, size=st.integers(0, 200))
def test_the_same_weights_spread_the_same_way_twice(weights, size) -> None:
    """A reference dataset whose composition moves is not a reference."""
    assert shares(weights, size) == shares(dict(reversed(list(weights.items()))), size)


# ── Dealing a topic's passages ───────────────────────────────────────────


@given(items=st.lists(st.integers(), max_size=40), wanted=st.integers(-2, 40))
def test_striding_drops_nothing_and_invents_nothing(items, wanted) -> None:
    """The first `wanted` are spread; the rest follow in their own order."""
    assert sorted(strided(list(items), wanted)) == sorted(items)


@given(items=st.lists(st.integers(), min_size=1, max_size=40))
def test_striding_is_a_permutation_of_its_input(items) -> None:
    """Read as a reordering, so a duplicate must survive as a duplicate."""
    strode = strided(list(items), len(items) // 2)

    assert len(strode) == len(items)


@given(share=st.floats(0.0, 1.0), over=st.integers(1, 200))
def test_a_share_picks_about_that_share_of_them(share, over) -> None:
    """Spread by position, so it picks one in every `1/share` exactly."""
    picked = sum(1 for index in range(over) if spread(index, share))

    assert abs(picked - over * share) <= 1


@given(over=st.integers(1, 50))
def test_a_share_of_none_picks_none_and_a_share_of_all_picks_all(over) -> None:
    """Both ends without a branch of their own."""
    assert not any(spread(index, 0.0) for index in range(over))
    assert all(spread(index, 1.0) for index in range(over))


# ── The band a question turns out to be ──────────────────────────────────


@given(
    passages=st.integers(1, 6),
    documents=st.integers(1, 6),
    topics=st.integers(0, 6),
    answer_chars=st.one_of(st.none(), st.integers(0, 2000)),
    follows=st.booleans(),
    long_answer=st.integers(1, 500),
)
def test_a_band_always_agrees_with_its_own_score(
    passages, documents, topics, answer_chars, follows, long_answer
) -> None:
    """A row disagreeing with its own score is what this exists to catch.

    `criteria_of` is read three times - by the service, by the writer and
    by a re-check - and the band has to be recomputable from the object.
    """
    read = criteria_of(
        passages=passages,
        documents=documents,
        topics=topics,
        answer_chars=answer_chars,
        follows=follows,
        long_answer=long_answer,
    )

    assert read.difficulty == band(read.score)


@given(score=st.integers(0, 5))
def test_a_band_never_falls_as_the_score_rises(score) -> None:
    """More of the things that make a question harder is never easier."""
    order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}

    assert order[band(score)] <= order[band(score + 1)]


# ── Drawing a release ────────────────────────────────────────────────────

_BANDS = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_TYPES = (QuestionType.FACTOID, QuestionType.REASON, QuestionType.COMPARISON)

POOL = st.lists(
    st.tuples(st.booleans(), st.sampled_from(_BANDS), st.sampled_from(_TYPES)),
    max_size=60,
)


@given(pool=POOL, size=st.integers(0, 40))
def test_a_draw_never_exceeds_the_size_it_was_asked_for(pool, size) -> None:
    """A release larger than its quota is a quota that decided nothing."""
    rows = [
        Row(id=at, answerable=a, difficulty=b, question_type=t)
        for at, (a, b, t) in enumerate(pool, 1)
    ]
    quota = quota_of(size, dict.fromkeys(_BANDS, 1), dict.fromkeys(_TYPES, 1), 0.3)

    assert choose(rows, quota).size <= size


@given(pool=POOL, size=st.integers(0, 40))
def test_a_draw_takes_each_question_at_most_once(pool, size) -> None:
    """A cell is popped from, so a row cannot fill two places."""
    rows = [
        Row(id=at, answerable=a, difficulty=b, question_type=t)
        for at, (a, b, t) in enumerate(pool, 1)
    ]
    quota = quota_of(size, dict.fromkeys(_BANDS, 1), dict.fromkeys(_TYPES, 1), 0.3)
    drawn = choose(rows, quota).ids

    assert len(drawn) == len(set(drawn))
    assert set(drawn) <= {row.id for row in rows}


@given(pool=POOL, size=st.integers(0, 40))
def test_a_draw_never_takes_more_unanswerable_than_the_ceiling(pool, size) -> None:
    """A ceiling and not a target: too many tests whether a chatbot says no."""
    rows = [
        Row(id=at, answerable=a, difficulty=b, question_type=t)
        for at, (a, b, t) in enumerate(pool, 1)
    ]
    quota = quota_of(size, dict.fromkeys(_BANDS, 1), dict.fromkeys(_TYPES, 1), 0.3)
    by_id = {row.id: row for row in rows}
    drawn = choose(rows, quota).ids

    assert sum(1 for one in drawn if not by_id[one].answerable) <= quota.unanswerable


@given(pool=POOL, size=st.integers(0, 40))
def test_the_same_pool_draws_the_same_release_twice(pool, size) -> None:
    """Ties go to the lowest id, so nothing here is order-dependent."""
    rows = [
        Row(id=at, answerable=a, difficulty=b, question_type=t)
        for at, (a, b, t) in enumerate(pool, 1)
    ]
    quota = quota_of(size, dict.fromkeys(_BANDS, 1), dict.fromkeys(_TYPES, 1), 0.3)

    assert choose(rows, quota).ids == choose(list(reversed(rows)), quota).ids


@given(size=st.integers(0, 200), share=st.floats(0.0, 1.0))
def test_a_quota_spends_every_place_it_has(size, share) -> None:
    """The bands and the kinds each account for the whole release."""
    quota = quota_of(size, dict.fromkeys(_BANDS, 1), dict.fromkeys(_TYPES, 1), share)

    assert sum(quota.difficulty.values()) == size
    assert sum(quota.types.values()) == size
    assert 0 <= quota.unanswerable <= size


def test_an_empty_quota_draws_an_empty_release() -> None:
    """A pool with nothing in it is not an error."""
    assert choose([], Quota(size=5, unanswerable=1, difficulty={}, types={})).ids == []


# ── Reading a number out of a sentence ───────────────────────────────────


@given(
    value=st.integers(0, 10**9),
    before=st.sampled_from(["", "There were ", "Total: ", "(", "about "]),
    after=st.sampled_from(["", ".", ",", " people.", ")", " in total"]),
)
def test_a_whole_number_is_read_wherever_a_sentence_puts_it(
    value, before, after
) -> None:
    """The gap this closes: 1200 in `1200.` was not a figure at all.

    A four-digit number before a full stop matched neither alternative of
    the pattern, so an aggregation over it was refused for finding no
    arithmetic.
    """
    assert float(value) in figures(f"{before}{value}{after}")


@given(value=st.integers(0, 999), group=st.integers(100, 999))
def test_a_thousands_group_is_one_number_in_either_convention(value, group) -> None:
    """1.234 and 1,234 are 1234, whichever locale wrote it."""
    expected = float(f"{value}{group:03d}")

    assert figures(f"{value}.{group:03d}") == [expected]
    assert figures(f"{value},{group:03d}") == [expected]


@given(text=st.text(max_size=200))
def test_reading_figures_never_raises_and_never_returns_a_nan(text) -> None:
    """It is handed passage text, which is whatever the corpus holds."""
    found = figures(text)

    assert not any(isnan(one) or isinf(one) for one in found), found
