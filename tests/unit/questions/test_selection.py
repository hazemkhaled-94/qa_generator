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

import pytest
from factories import bridge, resting, source

from database.qa_generator import (
    DocumentScope,
    FactKind,
    PassageScope,
    TopicScope,
)
from question_generation.models import FactGroup
from question_generation.selection import (
    Deal,
    Shape,
    by_passage,
    condensed_first,
    meets,
    overlap,
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


# ── A fact that already spans several passages ────────────────────────────


def test_a_bridge_is_offered_alone() -> None:
    """It already spans two passages, so a partner would put it over budget."""
    spanning = bridge(
        resting(2, document="doc-b"),
        fact_id=1,
        passage_id=1,
        document="doc-a",
    )
    deal = Deal([spanning, *facts_of(3, "doc-c")], wanted=4, size=4)

    sample = deal.sample(Shape.CROSS)

    assert sample is not None
    assert [fact.id for fact in sample.facts] == [1]
    assert len(sample.resting) == 2, "its own two passages, and no third"


def test_a_bridge_reads_as_the_spread_it_rests_on() -> None:
    """One fact, two passages, two documents: a cross-document question."""
    spanning = bridge(
        resting(2, document="doc-b", topic_id=9),
        passage_id=1,
        document="doc-a",
        topic_id=8,
    )

    read = FactGroup((spanning,)).criteria()

    assert read.passage_scope == PassageScope.MULTI
    assert read.document_scope == DocumentScope.CROSS
    assert read.topic_scope == TopicScope.MULTI


def test_the_verifier_is_shown_every_passage_a_bridge_rests_on() -> None:
    """Shown the anchor alone it could never recover the answer."""
    spanning = bridge(
        resting(2, document="doc-b", text="Urgent requests take 4 hours."),
        passage_id=1,
        passage_text="Standard requests take 48 hours.",
    )

    assert FactGroup((spanning,)).passages == (
        "Standard requests take 48 hours.",
        "Urgent requests take 4 hours.",
    )


def test_a_bridge_names_the_titles_of_both_its_documents() -> None:
    """The free leaks_source gate reads these, so it must see both."""
    spanning = bridge(
        resting(2, document="doc-b", document_title="The Urgent Handbook"),
        passage_id=1,
        document_title="The Standard Handbook",
    )

    assert FactGroup((spanning,)).titles == (
        "The Standard Handbook",
        "The Urgent Handbook",
    )


def test_a_passage_a_bridge_rests_on_is_never_dealt_again() -> None:
    """Asking about it twice would ask the same thing twice."""
    spanning = bridge(
        resting(2, document="doc-b"), fact_id=1, passage_id=1, document="doc-a"
    )
    deal = Deal([spanning, *facts_of(2, "doc-b")], wanted=4, size=2)

    first = deal.sample(Shape.SINGLE)
    second = deal.sample(Shape.SINGLE)

    assert first is not None and [fact.id for fact in first.facts] == [1]
    assert second is None, "passage 2 was already offered inside the bridge"


# ── More than one sample from one passage ──────────────────────────────────


def test_a_passage_is_offered_as_many_times_as_rounds_allow() -> None:
    """A passage carrying a dozen facts holds a dozen questions.

    Offering it once was the ceiling on the whole stage: 18.9% of one
    corpus's passages were ever read and 1.8% of its facts, because each
    topic stopped after its passages rather than after its material.
    """
    deal = Deal(facts_of(1, "a", count=6), wanted=10, size=2, rounds=3)

    dealt = [deal.sample() for _ in range(4)]

    assert [one is not None for one in dealt] == [True, True, True, False]


def test_no_fact_is_offered_twice_across_rounds() -> None:
    """What must not repeat is the material, not the passage it sits in."""
    deal = Deal(facts_of(1, "a", count=6), wanted=10, size=2, rounds=3)

    seen = []
    while (sample := deal.sample()) is not None:
        seen.extend(fact.id for fact in sample.facts)

    assert len(seen) == 6
    assert len(seen) == len(set(seen))


def test_a_passage_runs_out_of_facts_before_it_runs_out_of_rounds() -> None:
    """Rounds are a ceiling; the facts are the real limit.

    So a thin passage yields one sample and a dense one yields several,
    without either being configured.
    """
    deal = Deal(facts_of(1, "a", count=2), wanted=10, size=2, rounds=5)

    dealt = [deal.sample() for _ in range(3)]

    assert [one is not None for one in dealt] == [True, False, False]


def test_rounds_of_one_is_what_it_always_was() -> None:
    """The default, so nothing changes for a deployment that does not ask."""
    facts = [f for p in range(3) for f in facts_of(p, "a", count=4)]

    once = Deal(facts, wanted=10, size=2)
    dealt = [once.sample() for _ in range(4)]

    assert [one is not None for one in dealt] == [True, True, True, False]


def test_a_bridges_second_passage_is_used_up_whole() -> None:
    """It has been asked about once the bridge has, however many rounds remain."""
    spanning = bridge(resting(2, "b"), fact_id=99, passage_id=1, document="a")
    deal = Deal([spanning, *facts_of(2, "b", count=4)], wanted=10, size=2, rounds=3)

    first = deal.sample()

    assert first is not None
    assert [fact.id for fact in first.facts] == [spanning.id]
    assert deal.sample() is None


# ── Which kinds a sample offers ────────────────────────────────────────────


def test_the_kinds_of_a_passage_are_interleaved() -> None:
    """A sample capped at four spent all four on atomic claims.

    Which left the summary and the outline unoffered, and those two are what
    the types a single claim cannot answer are written from: a definition
    asks what something IS, an enumeration wants a real set.
    """
    facts = [
        source(1, passage_id=1, kind=FactKind.ATOMIC),
        source(2, passage_id=1, kind=FactKind.ATOMIC),
        source(3, passage_id=1, kind=FactKind.ATOMIC),
        source(4, passage_id=1, kind=FactKind.SUMMARY),
        source(5, passage_id=1, kind=FactKind.OUTLINE),
    ]

    assert [fact.kind for fact in ranked(facts)] == [
        FactKind.ATOMIC,
        FactKind.SUMMARY,
        FactKind.OUTLINE,
        FactKind.ATOMIC,
        FactKind.ATOMIC,
    ]


def test_a_value_still_comes_first_within_a_kind() -> None:
    """The two orderings are both applied, not one instead of the other."""
    facts = [
        source(1, passage_id=1, statement="It is reviewed regularly."),
        source(2, passage_id=1, statement="Reviewed every 4 years.", units=("4",)),
        source(3, passage_id=1, kind=FactKind.SUMMARY),
    ]

    assert [fact.id for fact in ranked(facts)] == [2, 3, 1]


def test_one_kind_is_left_in_its_own_order() -> None:
    """A corpus extracted with EXTRACTION_KINDS=atomic deals as it always did."""
    facts = facts_of(1, "a", count=4)

    assert [fact.id for fact in ranked(facts)] == [one.id for one in facts]


# ── Which facts a wide sample offers, which is what welds two into one ────


def test_a_wide_sample_offers_the_facts_the_two_passages_have_in_common() -> None:
    """The defect that produced two facts in a trenchcoat.

    Both passages are dealt, and each one's own best fact is its first. Taking
    those two independently paired a date with a category and the writer
    joined them with "and". The second passage's share is chosen for what it
    has in common with the first instead.
    """
    head = [
        source(
            1,
            document="a",
            passage_id=1,
            statement="Metamorphic testing is a technique.",
        ),
        source(
            2,
            document="a",
            passage_id=1,
            statement="Metamorphic testing was proposed in 1998.",
        ),
    ]
    partner = [
        # The two its own rank puts first - a fact carrying a value leads -
        # and neither is about anything the head is about.
        source(
            3,
            document="b",
            passage_id=2,
            statement="The northern site employs 40 people.",
            units=("40",),
        ),
        source(
            5,
            document="b",
            passage_id=2,
            statement="The southern site employs 25 people.",
            units=("25",),
        ),
        # Last by rank, and the only one the head meets.
        source(
            4,
            document="b",
            passage_id=2,
            statement="Metamorphic testing needs a source test case.",
        ),
    ]
    deal = Deal(head + partner, wanted=2, size=4)

    offered = deal.sample(Shape.CROSS)

    assert offered is not None
    chosen = {fact.id for fact in offered.facts}
    assert chosen >= {1, 2}, "the head passage was not offered whole"
    assert 4 in chosen, "the related fact was passed over for the passage's own best"


def test_the_first_passage_is_still_offered_in_rank_order() -> None:
    """Relatedness only decides the second side; there is nothing to relate to yet."""
    facts = [
        source(1, document="a", passage_id=1, statement="It is reviewed regularly."),
        source(
            2,
            document="a",
            passage_id=1,
            statement="Reviewed every 4 years.",
            units=("4",),
        ),
    ]
    deal = Deal(facts, wanted=1, size=2)

    offered = deal.sample(Shape.SINGLE)

    assert offered is not None
    assert [fact.id for fact in offered.facts] == [2, 1]


class TestPairingByCosine:
    """What pairs two passages, and two facts, once the corpus is embedded.

    A lemma overlap cannot see a synonym. Two passages about one subject in
    different words share a direction and no vocabulary, and Jaccard scores
    that pair 0 - so the sampler passed over it and paired each of them with
    something genuinely unrelated instead.
    """

    def test_two_facts_on_one_axis_meet(self) -> None:
        """Cosine 1, whatever words they are written in."""
        chosen = [
            source(fact_id=1, statement="Ein Prüffall deckt eine Klasse ab.", axis=3)
        ]
        candidate = source(fact_id=2, statement="A test case covers a class.", axis=3)

        assert meets(chosen, candidate) == pytest.approx(1.0)

    def test_two_facts_on_different_axes_do_not(self) -> None:
        """Orthogonal is as far apart as this space goes."""
        chosen = [source(fact_id=1, axis=3)]
        candidate = source(fact_id=2, axis=9)

        assert meets(chosen, candidate) == pytest.approx(0.0)

    def test_a_fact_sharing_no_word_still_meets_one_it_is_near(self) -> None:
        """The pair Jaccard scores 0 and cosine scores 1."""
        chosen = [source(fact_id=1, statement="Ein Fehlerzustand liegt vor.", axis=3)]
        candidate = source(fact_id=2, statement="A defect is present.", axis=3)

        assert meets(chosen, candidate) > 0.9
        assert not set(chosen[0].statement.split()) & set(candidate.statement.split())

    def test_a_candidate_is_weighed_against_the_nearest_chosen(self) -> None:
        """Not their mean, which points somewhere none of them is."""
        chosen = [source(fact_id=1, axis=3), source(fact_id=2, axis=9)]
        candidate = source(fact_id=3, axis=9)

        assert meets(chosen, candidate) == pytest.approx(1.0)

    def test_facts_with_no_vector_fall_back_to_their_lemmas(self) -> None:
        """A corpus extracted before the column existed still deals."""
        chosen = [source(fact_id=1, statement="The device weighs 4 kg.")]
        near = source(fact_id=2, statement="The device ships in March.")
        far = source(fact_id=3, statement="Refunds are paid within a week.")

        assert meets(chosen, near) > meets(chosen, far)

    def test_two_passages_on_one_axis_overlap(self) -> None:
        """The same measure, one level up."""
        left = [source(fact_id=1, passage_id=1, passage_axis=4)]
        right = [source(fact_id=2, passage_id=2, passage_axis=4)]

        assert overlap(left, right) == pytest.approx(1.0)

    def test_passages_with_no_vector_fall_back_to_their_lemmas(self) -> None:
        """Jaccard over what chunking stored, as before."""
        left = [source(fact_id=1, passage_id=1, lemmas=("test", "fall"))]
        right = [source(fact_id=2, passage_id=2, lemmas=("test", "fall"))]
        other = [source(fact_id=3, passage_id=3, lemmas=("rückgabe",))]

        assert overlap(left, right) == pytest.approx(1.0)
        assert overlap(left, other) == pytest.approx(0.0)


class Ordering:
    """A reranker that prefers whichever passage says a given word."""

    def __init__(self, wanted: str) -> None:
        """Initialises with the word that wins, and nothing asked yet."""
        self.wanted = wanted
        self.asked: list[list[str]] = []

    def ordered(self, query: str, documents, instruction: str = ""):
        """Scores 1 for a document carrying the word and 0 for the rest."""
        self.asked.append(list(documents))
        scored = [
            (position, 1.0 if self.wanted in document else 0.0)
            for position, document in enumerate(documents)
        ]
        return sorted(scored, key=lambda one: (-one[1], one[0]))


def test_a_reranker_reorders_the_shortlist_the_measure_narrowed() -> None:
    """The measure compares directions; the cross-encoder reads the pair.

    Both passages are candidates and the vectors cannot tell them apart -
    neither carries one here, so the lemma fallback scores both 0 and the
    lower passage id would win. The reranker is what picks the one that
    actually meets the head.
    """
    head = facts_of(1, "a", count=1)
    reranker = Ordering(wanted="Rückgabefrist")
    deal = Deal(
        head + facts_of(2, "b", count=1) + facts_of(3, "b", count=1),
        wanted=3,
        size=2,
        reranker=reranker,
    )

    deal.sample(Shape.CROSS)

    assert reranker.asked, "the reranker was never consulted"


def test_no_reranker_leaves_the_measure_in_charge() -> None:
    """A deployment that has named none gets exactly what it got before."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "b"), wanted=2, size=4)

    assert deal.sample(Shape.CROSS) is not None


# ── Widening a sample for the thread that follows it ───────────────────────


def test_a_widened_sample_reaches_facts_the_root_did_not() -> None:
    """The defect the follow-up gates were added against.

    `_followups` was handed the root's own sample, so the only material a
    thread could ask about was the material its first turn had already
    answered. Over one measured run 739 of 1,047 accepted follow-ups -
    70.6% - cited nothing new.
    """
    deal = Deal(facts_of(1, "a", count=4), wanted=1, size=2)
    root = deal.sample()
    assert root is not None

    wider = deal.widen(root)

    assert set(root.facts) < set(wider.facts), "the sample was not widened"


def test_widening_stays_on_the_passages_the_root_used() -> None:
    """A thread that changes material is not a conversation."""
    deal = Deal(facts_of(1, "a", count=4) + facts_of(2, "b", count=4), wanted=2, size=2)
    root = deal.sample()
    assert root is not None
    used = {passage.id for passage in root.resting}

    wider = deal.widen(root)

    assert {passage.id for passage in wider.resting} == used, (
        "widening reached a passage the root never used"
    )


def test_a_passage_with_nothing_left_widens_to_itself() -> None:
    """A thread that stops because the material ran out is the honest end.

    `asks_nothing_new` is what refuses the follow-up then; manufacturing a
    turn out of another passage would be the defect this avoids.
    """
    deal = Deal(facts_of(1, "a", count=2), wanted=1, size=2)
    root = deal.sample()
    assert root is not None

    assert deal.widen(root).facts == root.facts


def test_a_widened_fact_is_spent_and_never_dealt_again() -> None:
    """A sample is an offer, and offering one twice writes it twice."""
    deal = Deal(facts_of(1, "a", count=4) + facts_of(2, "b", count=2), wanted=2, size=2)
    root = deal.sample()
    assert root is not None
    taken = {fact.id for fact in deal.widen(root).facts}

    later = deal.sample()

    assert later is None or not (taken & {fact.id for fact in later.facts})


# ── Which shape of fact a form is offered first ────────────────────────────


def test_a_reading_is_offered_the_condensed_facts_first() -> None:
    """Aimed at the weld, and the sample size is why it works.

    At a cap of two over two passages each side offers ONE fact, and rank
    order makes that one atomic - so a cross-document sample was two single
    claims from two passages, which is the pair that has nothing between it.
    """
    facts = [
        source(1, passage_id=1, kind=FactKind.ATOMIC),
        source(2, passage_id=1, kind=FactKind.SUMMARY),
        source(3, passage_id=1, kind=FactKind.OUTLINE),
    ]
    deal = Deal(facts, wanted=1, size=1)

    offered = deal.sample(condensed=True)

    assert offered is not None
    assert offered.facts[0].kind in (FactKind.SUMMARY, FactKind.OUTLINE)


def test_a_value_is_still_offered_the_single_claims_first() -> None:
    """A factoid wants the fact carrying the number, not a paragraph."""
    facts = [
        source(1, passage_id=1, kind=FactKind.SUMMARY),
        source(2, passage_id=1, kind=FactKind.ATOMIC),
    ]
    deal = Deal(facts, wanted=1, size=1)

    offered = deal.sample(condensed=False)

    assert offered is not None
    assert offered.facts[0].kind == FactKind.ATOMIC


def test_reordering_keeps_the_order_ranked_put_each_half_in() -> None:
    """`ranked` puts the facts carrying units first; that has to survive."""
    facts = [
        source(1, passage_id=1, kind=FactKind.ATOMIC),
        source(2, passage_id=1, kind=FactKind.SUMMARY),
        source(3, passage_id=1, kind=FactKind.SUMMARY, units=("4 kg",)),
    ]

    moved = condensed_first(ranked(facts))

    assert [one.id for one in moved] == [3, 2, 1]


def test_a_passage_with_no_condensed_fact_is_offered_unchanged() -> None:
    """Most passages are mostly atomic, and none of them is refused for it."""
    facts = [source(n, passage_id=1, kind=FactKind.ATOMIC) for n in (1, 2, 3)]

    assert condensed_first(facts) == facts


# ── The floor under a second passage ───────────────────────────────────────


def test_no_floor_offers_a_second_passage_whatever_it_holds() -> None:
    """The default, and what every run so far has done."""
    deal = Deal(facts_of(1, "a") + facts_of(2, "b"), wanted=2, size=4, floor=0.0)

    offered = deal.sample(Shape.CROSS)

    assert offered is not None
    assert len(offered.resting) == 2


def test_a_floor_nothing_clears_narrows_to_one_passage() -> None:
    """The weld, narrowed for the writer instead of asked of it.

    The facts here share no vocabulary and carry no vectors, so `meets`
    scores the pair 0 and no floor above it can be cleared.
    """
    deal = Deal(
        facts_of(1, "a", statement="The device weighs 4 kg.")
        + facts_of(2, "b", statement="Quarterly dividends resumed in Lisbon."),
        wanted=2,
        size=4,
        floor=0.5,
    )

    offered = deal.sample(Shape.CROSS)

    assert offered is not None
    assert len(offered.resting) == 1, "a pair with nothing in common was offered"


def test_a_floor_a_related_passage_clears_still_offers_two() -> None:
    """The floor must not simply make every question single-passage."""
    shared = "The device weighs 4 kg and ships from the plant."
    deal = Deal(
        facts_of(1, "a", statement=shared) + facts_of(2, "b", statement=shared),
        wanted=2,
        size=4,
        floor=0.5,
    )

    offered = deal.sample(Shape.CROSS)

    assert offered is not None
    assert len(offered.resting) == 2
