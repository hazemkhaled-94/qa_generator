"""What the deal offers the writer, and how wide it is.

Every decision here is deterministic on purpose. A reference dataset whose
contents move between runs of the same corpus is not a reference, which is
the same reason TOPIC_RANDOM_STATE is fixed.

The measurement these exist to hold: on the corpus this was written against,
the deal that preceded them offered two passages in 5 of 214 samples, because
`size` flushed a group and then added the next passage whole. Almost every
question was single-passage, and `hard` was unreachable.
"""

from __future__ import annotations

from factories import source

from database.qa_generator import DocumentScope, PassageScope, TopicScope
from question_generation.selection import (
    Deal,
    Shape,
    by_passage,
    ranked,
    spread,
    strided,
)


def facts_of(passage_id: int, document: str, count: int = 2, **kwargs) -> list:
    """One passage's worth of facts, numbered so no two collide."""
    return [
        source(passage_id * 100 + n, document=document, passage_id=passage_id, **kwargs)
        for n in range(count)
    ]


# ── The cap, which is what was broken ─────────────────────────────────────


def test_a_sample_never_offers_more_facts_than_the_cap() -> None:
    """The defect that made every sample one passage.

    `size` used to flush the group and then add the next passage entire, so a
    passage of 37 facts was offered whole and nothing else fitted beside it.
    """
    deal = Deal(facts_of(1, "a", count=37), wanted=1, size=6)

    sample = deal.sample()

    assert sample is not None
    assert len(sample.facts) == 6


def test_a_wide_sample_offers_both_passages_rather_than_filling_from_one() -> None:
    """The cap is divided, so what is being compared is both there."""
    deal = Deal(
        facts_of(1, "a", count=10) + facts_of(2, "b", count=10), wanted=2, size=6
    )

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert {fact.passage_id for fact in sample.facts} == {1, 2}
    assert len(sample.facts) == 6


def test_a_passage_with_one_fact_still_offers_it() -> None:
    """Dividing the cap never rounds a passage down to nothing."""
    deal = Deal(facts_of(1, "a", count=1) + facts_of(2, "b", count=1), wanted=2, size=1)

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert len(sample.facts) == 2


# ── The shapes, and what each is worth ────────────────────────────────────


def test_a_single_sample_is_one_passage() -> None:
    """Which is the easy band: nothing above one."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "b"), wanted=2, size=4)

    sample = deal.sample(Shape.SINGLE)

    assert sample is not None
    assert sample.criteria().passage_scope == PassageScope.SINGLE
    assert sample.criteria().document_scope == DocumentScope.SINGLE


def test_a_cross_sample_reaches_another_document() -> None:
    """Two points: a cross-document question is multi-passage by construction."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "b"), wanted=2, size=4)

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert sample.criteria().document_scope == DocumentScope.CROSS
    assert sample.criteria().passage_scope == PassageScope.MULTI


def test_a_bridge_sample_reaches_another_document_and_another_subject() -> None:
    """Three points, which is the only way `hard` is reached by evidence."""
    deal = Deal(
        facts_of(1, "a", topic_id=7),
        facts_of(2, "b", topic_id=8),
        wanted=1,
        size=4,
    )

    sample = deal.sample(Shape.BRIDGE)

    assert sample is not None
    assert sample.criteria().topic_scope == TopicScope.MULTI
    assert sample.criteria().document_scope == DocumentScope.CROSS
    assert sample.criteria().difficulty == "hard"


def test_a_topic_in_one_document_falls_back_rather_than_writing_nothing() -> None:
    """A subject with no cross-document question in it is still a subject."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "a"), wanted=2, size=4)

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert sample.criteria().passage_scope == PassageScope.MULTI
    assert sample.criteria().document_scope == DocumentScope.SINGLE


def test_a_topic_of_one_passage_offers_it_alone() -> None:
    """Asking for a shape nothing can supply is not a reason to yield none."""
    deal = Deal(facts_of(1, "a"), wanted=1, size=4)

    sample = deal.sample(Shape.BRIDGE)

    assert sample is not None
    assert {fact.passage_id for fact in sample.facts} == {1}


def test_the_nearest_passage_of_a_document_is_the_one_paired() -> None:
    """Passages are numbered in reading order, so a neighbour is related."""
    deal = Deal(
        facts_of(1, "a", ordinal=1)
        + facts_of(2, "a", ordinal=2)
        + facts_of(9, "a", ordinal=40),
        wanted=1,
        size=4,
    )

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert {fact.passage_id for fact in sample.facts} == {1, 2}


def test_the_passage_sharing_most_vocabulary_is_the_one_paired() -> None:
    """Two passages about related things, as the corpus itself says."""
    deal = Deal(
        facts_of(1, "a", lemmas=("fee", "licence", "bank"))
        + facts_of(2, "b", lemmas=("weather", "rainfall"))
        + facts_of(3, "c", lemmas=("fee", "licence", "insurer")),
        wanted=1,
        size=6,
    )

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert {fact.passage_id for fact in sample.facts} == {1, 3}


# ── Coverage: what a run reaches ──────────────────────────────────────────


def test_a_passage_is_offered_once() -> None:
    """Two questions from one passage are two questions about one thing."""
    deal = Deal([f for p in range(6) for f in facts_of(p, "a")], wanted=6, size=4)

    seen = []
    while (sample := deal.sample()) is not None:
        seen.extend({fact.passage_id for fact in sample.facts})

    assert len(seen) == len(set(seen))


def test_a_pairing_never_offers_a_passage_a_later_sample_would_get() -> None:
    """A partner is used, so it is not dealt again as a head."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "b"), wanted=2, size=4)

    first = deal.sample(Shape.CROSS)

    assert first is not None
    assert deal.sample() is None


def test_the_deal_runs_out_rather_than_repeating() -> None:
    """A topic of three passages yields three questions, not twenty."""
    deal = Deal([f for p in range(3) for f in facts_of(p, "a")], wanted=20, size=4)

    dealt = [deal.sample() for _ in range(5)]

    assert [one is not None for one in dealt] == [True, True, True, False, False]


def test_passages_are_strided_over_the_whole_topic() -> None:
    """A topic of eighty passages asked about across all of it, not its start.

    Ten questions used to mean the first ten passages in document order, so
    two thirds of a big topic was never asked about at all.
    """
    reordered = strided(list(range(20)), 4)

    assert reordered[:4] == [0, 5, 10, 15]
    assert sorted(reordered) == list(range(20))


def test_striding_keeps_everything_when_more_is_wanted_than_there_is() -> None:
    """Nothing is dropped; the rest follow in their own order."""
    assert strided([1, 2, 3], 10) == [1, 2, 3]


# ── Determinism ───────────────────────────────────────────────────────────


def test_the_same_facts_deal_the_same_samples_whatever_order_they_arrive() -> None:
    """The dataset must not depend on what the database returned first."""
    facts = [f for p in range(6) for f in facts_of(p, "ab"[p % 2])]

    def dealt(rows):
        """Every sample one deal produces, as fact ids."""
        deal = Deal(rows, wanted=6, size=4)
        got = []
        while (sample := deal.sample(Shape.CROSS)) is not None:
            got.append([fact.id for fact in sample.facts])
        return got

    assert dealt(facts) == dealt(list(reversed(facts)))


def test_a_fact_carrying_a_value_is_offered_first() -> None:
    """A checkable question is written from a number, a date or an amount."""
    facts = [
        source(1, passage_id=1, statement="It is reviewed regularly."),
        source(
            2, passage_id=1, statement="It is reviewed every 4 years.", units=("4",)
        ),
    ]

    assert [fact.id for fact in ranked(facts)] == [2, 1]


def test_facts_are_grouped_by_the_passage_they_came_from() -> None:
    """A passage is the unit dealt, because its facts share a subject."""
    grouped = by_passage(facts_of(1, "a") + facts_of(2, "a"))

    assert [{fact.passage_id for fact in one} for one in grouped] == [{1}, {2}]


def test_no_facts_is_no_samples_rather_than_an_empty_one() -> None:
    """An empty group has no language and no difficulty to read."""
    assert Deal([], wanted=5, size=2).sample() is None


# ── The share spread every position-taken decision reads ──────────────────


def test_the_unanswerable_share_is_exact_and_spread_out() -> None:
    """A quarter means one in every four, not a coin weighted a quarter."""
    over = [spread(index, 0.25) for index in range(12)]

    assert sum(over) == 3
    assert over == [False, False, False, True] * 3


def test_a_share_of_zero_perturbs_nothing_and_a_share_of_one_perturbs_all() -> None:
    """Both ends without a branch of their own."""
    assert not any(spread(index, 0.0) for index in range(10))
    assert all(spread(index, 1.0) for index in range(10))
