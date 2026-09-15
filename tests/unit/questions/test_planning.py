"""What a topic is planned to be asked, before anything is written.

The plan is what makes the set configurable: a type with no weight is never
written, and a band is a request for a shape of sample rather than a verdict
about a question. Nothing here calls a model.
"""

from __future__ import annotations

from collections import Counter

from database.qa_generator import Difficulty, QuestionType
from question_generation.planning import allocate, plans
from question_generation.selection import Shape
from question_generation.types import SPECS

MIX = {
    QuestionType.FACTOID: 3,
    QuestionType.REASON: 2,
    QuestionType.COMPARISON: 1,
}

BANDS = {Difficulty.EASY: 2, Difficulty.MEDIUM: 2, Difficulty.HARD: 1}


def planned(**kwargs):
    """One topic's plan, with the mixes above unless a test says otherwise."""
    return plans(
        **{
            "wanted": 12,
            "types": MIX,
            "bands": BANDS,
            "unanswerable_share": 0.0,
            **kwargs,
        }
    )


# ── The allocator, which is what a mix means ──────────────────────────────


def test_the_weights_are_the_counts_over_a_whole_run() -> None:
    """3:2:1 over twelve slots is six, four and two."""
    counts = Counter(allocate(MIX, 12))

    assert counts == {
        QuestionType.FACTOID: 6,
        QuestionType.REASON: 4,
        QuestionType.COMPARISON: 2,
    }


def test_a_weight_of_zero_is_never_written() -> None:
    """Which is the switch for choosing what a run produces."""
    chosen = allocate({**MIX, QuestionType.REASON: 0}, 20)

    assert QuestionType.REASON not in chosen
    assert set(chosen) == {QuestionType.FACTOID, QuestionType.COMPARISON}


def test_the_types_are_interleaved_rather_than_run_in_blocks() -> None:
    """Other decisions are taken by position too.

    The unanswerable share and the follow-up share both read the slot number,
    so a mix run in blocks would always perturb the same type.
    """
    chosen = allocate(MIX, 12)

    assert chosen[:4] != [QuestionType.FACTOID] * 4
    assert len(set(chosen[:3])) > 1


def test_the_same_mix_allocates_the_same_slots_twice() -> None:
    """A reference dataset that moves between runs is not a reference."""
    assert allocate(MIX, 25) == allocate(MIX, 25)


def test_asking_for_nothing_allocates_nothing() -> None:
    """Zero slots and an empty mix are both answers, not failures."""
    assert allocate(MIX, 0) == []
    assert allocate({}, 10) == []


# ── The plan itself ───────────────────────────────────────────────────────


def test_every_slot_asked_for_is_planned() -> None:
    """QUESTIONS_PER_TOPIC is how many questions a topic is asked."""
    assert len(planned(wanted=12)) == 12


def test_each_band_gets_the_shape_that_makes_it_reachable() -> None:
    """A question drawn from one passage cannot be cross-document."""
    shapes = {one.band: one.shape for one in planned(wanted=30)}

    assert shapes[Difficulty.EASY] == Shape.SINGLE
    assert shapes[Difficulty.MEDIUM] == Shape.CROSS
    assert shapes[Difficulty.HARD] == Shape.BRIDGE


def test_a_type_needing_two_passages_is_never_planned_easy() -> None:
    """A comparison drawn from one passage is a question about one thing."""
    spanning = [
        one for one in planned(wanted=30, unanswerable_share=0.0) if one.spec.spans
    ]

    assert spanning
    assert all(one.band != Difficulty.EASY for one in spanning)
    assert all(one.shape != Shape.SINGLE for one in spanning)


def test_an_unanswerable_question_is_planned_easy_and_from_one_passage() -> None:
    """It is written by moving one fact out of reach.

    A second passage has nothing to do with that, and a chatbot declining a
    question is not made cleverer by the question having spanned two files.
    """
    unanswerable = [one for one in planned(unanswerable_share=1.0)]

    assert unanswerable
    assert all(one.shape == Shape.SINGLE for one in unanswerable)
    assert all(one.band == Difficulty.EASY for one in unanswerable)


def test_the_unanswerable_share_lands_on_more_than_one_type() -> None:
    """The aliasing an interleaved allocation exists to prevent."""
    perturbed = {
        one.spec.name
        for one in planned(wanted=24, unanswerable_share=0.25)
        if not one.answerable
    }

    assert len(perturbed) > 1


def test_a_plan_says_whether_the_writer_is_told_to_span() -> None:
    """The instruction follows the sample, not the type.

    A factoid drawn from two documents is a cross-document factoid, and the
    writer has to be told to use both halves or it answers the first.
    """
    wide = next(one for one in planned(wanted=30) if one.shape != Shape.SINGLE)
    narrow = next(one for one in planned(wanted=30) if one.shape == Shape.SINGLE)

    assert wide.spans
    assert not narrow.spans
    assert "MORE THAN ONE" in wide.spec.system(spans=wide.spans)
    assert "MORE THAN ONE" not in SPECS[QuestionType.FACTOID].system()


def test_the_same_settings_plan_the_same_topic_twice() -> None:
    """Two runs of one corpus have to produce one dataset."""
    first = [(one.spec.name, one.band, one.shape) for one in planned()]
    second = [(one.spec.name, one.band, one.shape) for one in planned()]

    assert first == second
