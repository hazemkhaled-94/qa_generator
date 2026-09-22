"""The ceilings the code already names, as budgets rather than comments.

Three places in this pipeline say in a comment how they do not scale:
`adds_up` searches every combination of up to four figures, `composition`
walks the pool once per release, and `shares` and `choose` run the draw.
Each is fine at the size this corpus is and each has a number in it that a
later change could raise without anybody measuring what that costs.

The budgets are wall clock and deliberately loose - several times the
measured figure - so they fail on a change of order and not on a slow
laptop. What they are protecting is the shape, not the constant.

Never on a pull request: `make test-perf` and the nightly run.
"""

from __future__ import annotations

import pytest

from question_generation.balance import Row, choose, composition, quota_of, shares
from question_generation.gates import _FIGURE_CAP, _TERM_CAP, adds_up

pytestmark = pytest.mark.perf

#: Seconds one call may take. Measured at roughly a fiftieth of this.
BUDGET = 2.0


def mean(benchmark) -> float:
    """The mean time one round took, in seconds."""
    return float(benchmark.stats.stats.mean)


def test_the_arithmetic_search_stays_inside_its_cap(benchmark) -> None:
    """Every combination of up to four of forty figures.

    `_FIGURE_CAP` and `_TERM_CAP` are what keep this from being
    exponential. Raising either is the change this measures.
    """
    passages = [" ".join(str(one) for one in range(1, _FIGURE_CAP + 20))]

    # A total no combination reaches, so the search runs to the end.
    answered = benchmark(adds_up, "999999", passages)

    assert answered is False
    assert mean(benchmark) < BUDGET, (
        f"the sum search is over budget; _FIGURE_CAP is {_FIGURE_CAP} and "
        f"_TERM_CAP is {_TERM_CAP}, and the search is every combination of "
        f"up to the second from the first"
    )


def test_reporting_a_release_walks_the_pool_once(benchmark) -> None:
    """It rebuilt the chosen set once per row of the pool.

    Quadratic in the pool, which at a thousand accepted questions is a
    million set builds for a report nobody waits for on purpose.
    """
    pool = [
        Row(
            id=at,
            answerable=at % 3 == 0,
            difficulty=("easy", "medium", "hard")[at % 3],
            question_type=("factoid", "reason", "comparison")[at % 3],
        )
        for at in range(1, 3001)
    ]
    ids = [row.id for row in pool[::2]]

    counted = benchmark(composition, pool, ids)

    assert sum(counted["difficulty"].values()) == len(ids)
    assert mean(benchmark) < BUDGET


def test_the_draw_scales_to_a_release_worth_having(benchmark) -> None:
    """Largest deficit first, over every cell, once per place drawn."""
    bands = {"easy": 1, "medium": 1, "hard": 1}
    types = dict.fromkeys(
        ("factoid", "reason", "comparison", "definition", "entity"), 1
    )
    pool = [
        Row(
            id=at,
            answerable=at % 4 != 0,
            difficulty=("easy", "medium", "hard")[at % 3],
            question_type=list(types)[at % len(types)],
        )
        for at in range(1, 2001)
    ]

    drawn = benchmark(choose, pool, quota_of(900, bands, types, 0.2))

    assert drawn.size > 0
    assert mean(benchmark) < BUDGET


def test_spreading_a_size_over_its_weights_is_cheap(benchmark) -> None:
    """Called once per draw, and the draw is bisected for."""
    weights = {f"kind-{at}": at % 5 + 1 for at in range(40)}

    spread = benchmark(shares, weights, 5000)

    assert sum(spread.values()) == 5000
    assert mean(benchmark) < BUDGET
