"""Every gate, and the order they are applied in.

The order is not a detail. Each gate that fires saves the cost of the ones
behind it, and the only expensive one is last, so a test that lets a
malformed question reach the verifier is a test of a bill.
"""

from __future__ import annotations

import pytest
from factories import candidate, group, source

from database.qa_generator import QuestionRejection, QuestionStatus
from question_generation.verification import (
    Neighbour,
    QuestionChecker,
    agrees,
    near_verdict,
    structural,
)

pytestmark = pytest.mark.nlp


def code(failed) -> str | None:
    """The gate's code out of a verdict, or None when it passed."""
    return failed[0] if failed else None


def checked(**kwargs):
    """Runs the free gates over one candidate's fields."""
    return structural(
        **{
            "question_text": "What does the device weigh?",
            "target_answer": "4 kg",
            "answerable": True,
            "language": "en",
            "statements": (),
            **kwargs,
        }
    )


# ── The free gates ─────────────────────────────────────────────────────────


def test_a_well_formed_question_passes() -> None:
    """The case every rejection below is measured against."""
    assert checked() is None


@pytest.mark.parametrize(
    ("why", "fields"),
    [
        ("nothing at all", {"question_text": "   "}),
        ("not a question", {"question_text": "The device weighs 4 kg."}),
        ("no target answer", {"target_answer": None}),
        ("a blank target answer", {"target_answer": "  "}),
        (
            "an unanswerable one with an answer",
            {"answerable": False, "target_answer": "4 kg"},
        ),
    ],
)
def test_what_counts_as_malformed(why: str, fields: dict) -> None:
    """Five ways of being unusable, under one code."""
    assert code(checked(**fields)) == QuestionRejection.MALFORMED, why


def test_a_fact_handed_straight_back_is_not_a_question() -> None:
    """The laziest thing a model can do with the prompt."""
    failed = checked(
        question_text="The device weighs 4 kg?",
        statements=("The device weighs 4 kg.",),
    )

    assert code(failed) == QuestionRejection.MALFORMED


def test_an_unanswerable_question_needs_no_target_answer() -> None:
    """It is scored on behaviour, which is what the table's CHECK says."""
    assert checked(answerable=False, target_answer=None) is None


def test_a_question_in_the_wrong_language_is_refused() -> None:
    """A German corpus answering English questions measures a translator."""
    failed = checked(
        question_text=(
            "Within how many hours is a standard support request answered "
            "on an ordinary working day?"
        ),
        target_answer="within 48 hours of the request being raised",
        language="de",
    )

    assert code(failed) == QuestionRejection.MALFORMED


def test_a_question_too_short_to_read_is_not_called_the_wrong_language() -> None:
    """Lingua answers nothing below forty characters of prose.

    Most questions are shorter than that, so a gate that treated `cannot
    tell` as `wrong` would reject almost everything.
    """
    assert (
        checked(question_text="Wie schwer?", target_answer="4 kg", language="de")
        is None
    )


# ── The probe ──────────────────────────────────────────────────────────────


def test_nothing_near_enough_is_not_a_duplicate() -> None:
    """The threshold is what makes the gate a gate."""
    near = Neighbour("Something else entirely?", answerable=True, similarity=0.5)

    assert near_verdict(near, answerable=True, threshold=0.93) is None


def test_an_empty_corpus_of_questions_rejects_nothing() -> None:
    """The first question of a run has nothing to be a duplicate of."""
    assert near_verdict(None, answerable=True, threshold=0.93) is None


def test_a_near_twin_of_an_accepted_question_is_a_duplicate() -> None:
    """A benchmark that asks the same thing twice weights it twice."""
    near = Neighbour("What is the weight?", answerable=True, similarity=0.97)

    assert code(near_verdict(near, answerable=True, threshold=0.93)) == (
        QuestionRejection.DUPLICATE
    )


def test_an_unanswerable_twin_of_an_answered_question_is_answerable_after_all() -> None:
    """One probe, two gates.

    If the corpus already answers a question this close, this one is not
    unanswerable, whatever the perturbation prompt intended.
    """
    near = Neighbour("What does the device weigh?", answerable=True, similarity=0.96)

    assert code(near_verdict(near, answerable=False, threshold=0.93)) == (
        QuestionRejection.ANSWERABLE_AFTER_ALL
    )


def test_two_unanswerable_near_twins_are_only_duplicates() -> None:
    """Neither of them is evidence the corpus answers anything."""
    near = Neighbour("What about on Mars?", answerable=False, similarity=0.96)

    assert code(near_verdict(near, answerable=False, threshold=0.93)) == (
        QuestionRejection.DUPLICATE
    )


# ── Agreement between the verifier and the target ──────────────────────────


def test_the_verifier_agrees_when_it_recovered_every_unit_of_the_target() -> None:
    """Wording is free to differ; the numbers and names are not."""
    assert agrees("48 hours", "within 48 hours", "en")


def test_the_verifier_disagrees_on_a_different_number() -> None:
    """The failure an embedding comparison lets through.

    `4 hours` and `48 hours` are close in every vector space and are not the
    same answer.
    """
    assert not agrees("4 hours", "48 hours", "en")


def test_a_target_with_no_units_falls_back_to_containment() -> None:
    """Not every answer is a number or a name."""
    assert agrees("The board is responsible.", "the board", "en")
    assert not agrees("The auditor is responsible.", "the board", "en")


# ── The gates in order ─────────────────────────────────────────────────────


class Recording:
    """An embedder and a verifier that count how often they were asked."""

    def __init__(
        self, vector: list[float] | None = None, recovers: str | None = "4 kg"
    ):
        """Initialises with what to answer, and nothing asked yet."""
        self.vector = vector or [1.0] + [0.0] * 1023
        self.recovers = recovers
        self.embedded = 0
        self.verified = 0

    def embed(self, text: str) -> list[float]:
        """Answers with the one vector, and counts the ask."""
        self.embedded += 1
        return self.vector

    def recover(self, question: str, passages) -> str | None:
        """Answers with the scripted recovery, and counts the ask."""
        self.verified += 1
        return self.recovers


def build(recording: Recording, near: Neighbour | None = None) -> QuestionChecker:
    """A checker over a scripted probe and a scripted pair of models."""
    return QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: near,
        threshold=0.93,
    )


def test_a_question_that_clears_every_gate_is_accepted() -> None:
    """With its embedding kept, so a later run can probe against it."""
    recording = Recording(recovers="4 kg")

    result = build(recording).check(candidate())

    assert result.accepted
    assert result.status == QuestionStatus.ACCEPTED
    assert result.embedding == recording.vector


def test_a_malformed_question_never_reaches_the_embedder_or_the_verifier() -> None:
    """The free gate is first because the two behind it are not free."""
    recording = Recording()

    result = build(recording).check(candidate(question_text="not a question"))

    assert result.rejected_reason == QuestionRejection.MALFORMED
    assert (recording.embedded, recording.verified) == (0, 0)


def test_a_duplicate_never_reaches_the_verifier() -> None:
    """An index probe is cheap and a model call is not."""
    recording = Recording()
    near = Neighbour("What is the weight?", answerable=True, similarity=0.99)

    result = build(recording, near).check(candidate())

    assert result.rejected_reason == QuestionRejection.DUPLICATE
    assert recording.verified == 0


def test_an_answer_the_passages_do_not_give_is_not_recoverable() -> None:
    """The gate no similarity measure makes."""
    recording = Recording(recovers=None)

    result = build(recording).check(candidate())

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE


def test_an_answer_that_contradicts_the_target_is_not_recoverable() -> None:
    """Recovering something is not recovering the right thing."""
    recording = Recording(recovers="12 hours")

    result = build(recording).check(candidate(target_answer="4 kg"))

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE


def test_an_unanswerable_question_the_passages_answer_is_rejected() -> None:
    """The round trip run the other way round."""
    recording = Recording(recovers="4 kg")

    result = build(recording).check(
        candidate(
            question_text="What does the device weigh?",
            target_answer=None,
            answerable=False,
        )
    )

    assert result.rejected_reason == QuestionRejection.ANSWERABLE_AFTER_ALL


def test_an_unanswerable_question_the_passages_do_not_answer_is_accepted() -> None:
    """Which is the one this whole kind of question exists for."""
    recording = Recording(recovers=None)

    result = build(recording).check(
        candidate(
            question_text="What does the device weigh on Mars?",
            target_answer=None,
            answerable=False,
        )
    )

    assert result.accepted


def test_a_duplicate_of_something_written_earlier_in_the_same_run_is_caught() -> None:
    """Nothing is stored until the topic is finished.

    Without this a topic would write the same question ten times and the
    database probe would see none of them.
    """
    recording = Recording()
    checker = build(recording)
    first = checker.check(candidate())

    second = checker.check(
        candidate(question_text="What is the device's weight?"), [first]
    )

    assert first.accepted
    assert second.rejected_reason == QuestionRejection.DUPLICATE


def test_the_verifier_is_shown_the_cited_passages_and_nothing_else() -> None:
    """The dataset is what is being measured, not a retriever."""
    seen: list = []

    class Watching(Recording):
        """Records what the verifier was shown."""

        def recover(self, question: str, passages) -> str | None:
            """Keeps the passages and answers as scripted."""
            seen.append(list(passages))
            return super().recover(question, passages)

    pair = group(
        source(1, document="a", passage_id=1, statement="The device weighs 4 kg."),
        source(2, document="b", passage_id=2, statement="The device ships in March."),
    )
    build(Watching()).check(candidate(facts=pair))

    assert len(seen[0]) == 2, "one passage per distinct citation"


def test_the_difficulty_is_read_off_the_group_rather_than_judged() -> None:
    """Two readers cannot disagree about it, which is the point."""
    pair = group(
        source(1, document="a", passage_id=1), source(2, document="b", passage_id=2)
    )

    result = build(Recording()).check(candidate(facts=pair))

    assert result.difficulty == "cross_document"
    assert result.fact_ids == (1, 2)
