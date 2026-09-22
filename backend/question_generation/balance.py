"""Choosing a balanced release out of everything a run accepted.

The gates decide whether a question is sound. Nothing before this decides
what the SET looks like, and the two are different problems: over one corpus
every gate did its job and the accepted set still came out 45% unanswerable
and 96.5% easy, because what survives a filter is whatever the material
happened to offer.

So the composition is chosen here instead, out of a pool deliberately larger
than the release. Three marginals are held at once - how many questions have
no answer, how they spread over the difficulty bands, and how they spread
over the kinds - and they are marginals rather than a joint distribution on
purpose. Demanding an even spread of kinds WITHIN each band is not reachable
and not wanted: a `factoid` answers with a value, a value is short, and a
short answer cannot earn the length point that a `hard` question generally
needs. Asking for both marginals is achievable; asking for their product is
asking the corpus to be something it is not.

Nothing is deleted and nothing is rewritten. A release is a column on the
rows already there, so the questions left out stay queryable and a second
run with different shares replaces the first.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Row:
    """One accepted question, as the choosing reads it."""

    id: int
    answerable: bool
    difficulty: str
    question_type: str


@dataclass(frozen=True)
class Quota:
    """How many of each marginal one release wants.

    `unanswerable` is a ceiling and the rest are targets. That asymmetry is
    the point of the setting: a release with too few unanswerable questions
    tests a little less than it could, and one with too many is mostly
    testing whether a chatbot can say no.
    """

    size: int
    unanswerable: int
    difficulty: dict[str, int]
    types: dict[str, int]


@dataclass(frozen=True)
class Release:
    """What was chosen, and what the quotas could not find."""

    ids: list[int]
    quota: Quota
    #: Bucket name to how many of it were missing, empty when all were met.
    short: dict[str, int]

    @property
    def size(self) -> int:
        """How many questions are in it."""
        return len(self.ids)


def shares(weights: Mapping[str, int], size: int) -> dict[str, int]:
    """Spreads `size` over named weights, summing to exactly `size`.

    Largest remainder: everybody gets their whole part, and the seats left
    over go to the largest fractions. Ties by name, so the same weights over
    the same size give the same answer twice.
    """
    total = sum(max(weight, 0) for weight in weights.values())
    if total <= 0 or size <= 0:
        return dict.fromkeys(weights, 0)
    exact = {name: size * max(weight, 0) / total for name, weight in weights.items()}
    whole = {name: int(value) for name, value in exact.items()}
    order = sorted(exact, key=lambda name: (whole[name] - exact[name], name))
    for name in order[: size - sum(whole.values())]:
        whole[name] += 1
    return whole


def quota_of(
    size: int,
    bands: Mapping[str, int],
    types: Mapping[str, int],
    unanswerable: float,
) -> Quota:
    """The quota a release of this size wants."""
    return Quota(
        size=size,
        unanswerable=int(size * unanswerable),
        difficulty=shares(bands, size),
        types=shares(types, size),
    )


def _celled(pool: Sequence[Row]) -> dict[tuple[bool, str, str], list[Row]]:
    """The pool grouped by the cell a question would spend its places in.

    Each in id order, so the same pool draws the same release twice.
    """
    held: dict[tuple[bool, str, str], list[Row]] = {}
    for row in sorted(pool, key=lambda one: one.id):
        held.setdefault((row.answerable, row.difficulty, row.question_type), []).append(
            row
        )
    return held


def choose(pool: Sequence[Row], quota: Quota) -> Release:
    """The questions filling this quota, and whatever it could not fill.

    Largest deficit first. A question spends a place in three buckets at
    once - answerability, band and kind - so at each step the cell taken
    from is the one whose band and kind are the furthest, as a SHARE of
    what each was asked for, from being filled. Measured as a share and not
    as a count: a band's quota is three times a kind's, so compared raw the
    band term decided every step and the kinds ran out underneath it.

    That keeps the marginals in step for almost all of the draw and not for
    the end of it. The last two percent is where three marginals over
    eleven kinds cannot all reach zero together, and a pool filling 98% of
    a release of 900 is not a pool that should be refused - which is what
    `TOLERANCE` is for, in `largest` above. This reports what it could not
    fill and leaves that judgement there.

    Ties go to the lowest question id, so the draw is reproducible.
    """
    bands = dict(quota.difficulty)
    types = dict(quota.types)
    left = {True: quota.size - quota.unanswerable, False: quota.unanswerable}
    cells = _celled(pool)
    chosen: list[int] = []

    while len(chosen) < quota.size:
        best: tuple[tuple[float, int], tuple[bool, str, str]] | None = None
        for key, rows in cells.items():
            answerable, band, kind = key
            if not rows or left[answerable] <= 0:
                continue
            if bands.get(band, 0) <= 0 or types.get(kind, 0) <= 0:
                continue
            # The SHARE of each bucket still to be filled, not the count.
            # Compared raw, a band's quota is three times a kind's and
            # eleven times nothing, so the band term decided every step
            # and the kinds ran out under it.
            score = (
                bands[band] / max(quota.difficulty.get(band, 1), 1)
                + types[kind] / max(quota.types.get(kind, 1), 1),
                -rows[0].id,
            )
            if best is None or score > best[0]:
                best = (score, key)
        if best is None:
            break
        answerable, band, kind = best[1]
        chosen.append(cells[best[1]].pop(0).id)
        left[answerable] -= 1
        bands[band] -= 1
        types[kind] -= 1

    short = {name: count for name, count in (bands | types).items() if count > 0}
    if left[True] > 0:
        short["answerable"] = left[True]
    return Release(ids=sorted(chosen), quota=quota, short=short)


#: How far short of its own quota one bucket may come and still be taken,
#: as a share of that quota. The target is "about a third each", not exactly
#: a third: three marginals over eleven kinds cannot all land on zero
#: together out of a finite pool, and the last few places are where it
#: always fails.
#:
#: The size of that endgame is what this has to be set against, and it does
#: not shrink as the pool grows - the draw runs about 5% short on the bands
#: at every size measured, 2 of 33 at one hundred and 17 of 333 at a
#: thousand. Calibrated at 3% against a pool of 235, it then refused a pool
#: of 2,259: the largest release it would take was 58, where 8% took 1,145
#: of the same questions with the bands closer to even and the kinds inside
#: 0.6 of a point. A tolerance below the endgame does not buy a tighter
#: release, it buys almost no release.
#:
#: Still far below what a real skew looks like. The draw this exists to
#: refuse was six aggregations short of fourteen - 43% of that bucket, and
#: that kind at 5.6% where an even mix wanted 9.1%.
TOLERANCE = 0.10


def close_enough(release: Release, tolerance: float = TOLERANCE) -> bool:
    """Whether every bucket came within its slack of what was asked for.

    Measured per bucket, not on the total, because the two say opposite
    things about the same draw. Over 727 accepted questions a release of
    143 was six aggregations short - that kind at 5.6% where the quota
    wanted 9.1%, which is the one thing this whole stage exists to prevent -
    and a release of 88 was short by exactly one in five buckets, which is
    the balanced set the pool can supply. Judged on the total the first
    looks like the better draw, and the bisection took neither: held to 3%
    of the total it refused 85 out of 88 and collapsed to 4.

    One place is always allowed. A quota of eight cannot absorb a
    proportional tolerance, and being one short of eight is where the
    rounding fell rather than a pool that cannot supply the kind.
    """
    wanted = {
        **release.quota.difficulty,
        **release.quota.types,
        "answerable": release.quota.size - release.quota.unanswerable,
    }
    return all(
        short <= max(1, tolerance * wanted.get(name, 0))
        for name, short in release.short.items()
    )


def largest(
    pool: Sequence[Row],
    bands: Mapping[str, int],
    types: Mapping[str, int],
    unanswerable: float,
    tolerance: float = TOLERANCE,
) -> Release:
    """The biggest release this pool fills, near enough, found by bisection.

    The size a balanced set can reach is set by whichever bucket is furthest
    from being able to supply its share: a pool that is 12% hard fills no
    release that wants a third hard beyond 12/33 of its own size. Rather
    than working that bound out for three interacting marginals, the
    choosing itself is the test, and the largest size it fills to within
    `tolerance` is bisected for.

    What the release actually came out as is reported rather than assumed -
    `composition` reads it off the rows - so the few places a bucket ends
    short are visible instead of being promised away.

    When no size fills, the largest draw attempted is handed back with its
    shortfall rather than nothing at all. A thin pool has an empty cell
    somewhere, one empty cell refuses every size down to one, and returning
    nothing for that reason tells a reader less than returning the best set
    available beside a note of what is missing from it: 29 accepted
    questions over eleven kinds drew zero, which is true to the quota and
    useless to anybody.
    """
    low, high = 1, len(pool)
    empty = Release(ids=[], quota=quota_of(0, bands, types, unanswerable), short={})
    best, most = empty, empty
    while low <= high:
        size = (low + high) // 2
        release = choose(pool, quota_of(size, bands, types, unanswerable))
        if release.size > most.size:
            most = release
        if close_enough(release, tolerance):
            best, low = release, size + 1
        else:
            high = size - 1
    return best if best.ids else most


def composition(pool: Sequence[Row], ids: Sequence[int]) -> dict[str, dict[str, int]]:
    """What a chosen set actually came out as, for the report."""
    chosen = set(ids)
    taken = [row for row in pool if row.id in chosen]
    return {
        "answerable": Counter(
            "unanswerable" if not row.answerable else "answerable" for row in taken
        ),
        "difficulty": Counter(row.difficulty for row in taken),
        "type": Counter(row.question_type for row in taken),
    }
