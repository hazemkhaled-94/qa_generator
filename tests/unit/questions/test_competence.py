"""The types whose answers are reasoned to rather than found.

Every other kind of question here is answered by a sentence in the corpus -
the recoverability gate refuses one whose answer is not - so however much
reading it takes, answering is retrieval. `implication` and `application`
are not: the premises are in the material and the conclusion is not, which
is what makes a cognitive level a measurement rather than a label.
"""

from __future__ import annotations

import pytest
from factories import candidate, group, source

from database.qa_generator import (
    CognitiveLevel,
    Derivation,
    QuestionRejection,
    QuestionType,
)
from question_generation.types import SPECS
from question_generation.verifier import Reading

pytestmark = pytest.mark.nlp


# ── What a type declares ───────────────────────────────────────────────────


def test_every_type_declares_a_level() -> None:
    """The column is written from this, so a type without one writes NULL."""
    missing = [name for name, spec in SPECS.items() if not spec.level]

    assert missing == []


def test_every_level_is_one_a_question_may_carry() -> None:
    """The CHECK on the column is the same list."""
    unknown = {spec.level for spec in SPECS.values()} - set(CognitiveLevel)

    assert unknown == set()


@pytest.mark.parametrize(
    ("kind", "level"),
    [
        (QuestionType.FACTOID, CognitiveLevel.RECALL),
        (QuestionType.DEFINITION, CognitiveLevel.UNDERSTAND),
        (QuestionType.CONDITION, CognitiveLevel.APPLY),
        (QuestionType.APPLICATION, CognitiveLevel.APPLY),
        (QuestionType.REASON, CognitiveLevel.ANALYSE),
        (QuestionType.IMPLICATION, CognitiveLevel.ANALYSE),
    ],
)
def test_a_type_asks_what_its_level_says(kind: str, level: str) -> None:
    """Pinned, because the level is read off the type and nothing else."""
    assert SPECS[kind].level == level


def test_only_the_derived_types_are_derived() -> None:
    """A derived type skips recoverability, so this list is load-bearing.

    Marking a retrieval type derived would send its questions to a gate
    that asks whether the answer FOLLOWS, and never check that the corpus
    actually contains it.
    """
    derived = {name for name, spec in SPECS.items() if spec.derived}

    assert derived == {
        QuestionType.AGGREGATION,
        QuestionType.IMPLICATION,
        QuestionType.APPLICATION,
    }


def test_arithmetic_and_entailment_are_told_apart() -> None:
    """They are checked by different prompts and must not share one.

    `Do the arithmetic` is the wrong instruction for a conclusion drawn
    from two rules, and `does this follow` is the wrong one for a total,
    which follows from anything if the reader is generous about addition.
    """
    assert SPECS[QuestionType.AGGREGATION].derived == Derivation.ARITHMETIC
    assert SPECS[QuestionType.IMPLICATION].derived == Derivation.ENTAILMENT
    assert SPECS[QuestionType.APPLICATION].derived == Derivation.ENTAILMENT


def test_an_implication_needs_two_passages() -> None:
    """A conclusion from one statement is that statement."""
    assert SPECS[QuestionType.IMPLICATION].spans


# ── Which gate a derived question faces ────────────────────────────────────


class Recording:
    """A verifier that records which question it was asked."""

    def __init__(self, answer: bool = True) -> None:
        """Initialises with the verdict it will give."""
        self.answer = answer
        self.asked: list[str] = []
        self.vector = [1.0] + [0.0] * 1023

    def embed(self, text: str) -> list[float]:
        """The one vector."""
        return self.vector

    def read(self, question, passages, thread=()):
        """Recall, which a derived question must never reach."""
        self.asked.append("read")
        return Reading(recovered=None)

    def names_its_source(self, question: str) -> bool:
        """The phrasing residue, which none of these questions trips."""
        return False

    def names_something(self, question: str) -> bool:
        """These questions all name something, thin though some read."""
        return True

    def self_contained(self, question: str, pointers) -> bool:
        """The other phrasing residue, asked only where a pointer was found."""
        return True

    def computes(self, question, answer, passages, thread=()) -> bool:
        """The arithmetic gate."""
        self.asked.append("computes")
        return self.answer

    def follows(self, question, answer, passages, thread=()) -> bool:
        """The entailment gate."""
        self.asked.append("follows")
        return self.answer

    def supports(self, question, answer, passages, thread=()) -> bool:
        """The rescue pass, which a derived question does not use."""
        self.asked.append("supports")
        return self.answer


def build(recording: Recording):
    """A checker over the scripted verifier."""
    from question_generation.checker import QuestionChecker

    return QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        bounds={"value": (1, 80), "list": (3, 300), "explanation": (20, 600)},
        phrasing=recording,
    )


#: An answer of the shape each derived type asks for. A total is a value
#: and is held to 80 characters; a conclusion is an explanation and must
#: carry a verb, so one answer cannot serve both.
_ANSWERS = {
    QuestionType.AGGREGATION: "65",
    QuestionType.IMPLICATION: (
        "the following Tuesday, because the hours count only working days"
    ),
    QuestionType.APPLICATION: (
        "within four hours, because work has stopped at the site"
    ),
}


def reasoned(kind: str):
    """One derived question, with an answer the passages do not state."""
    return candidate(
        question_text="A site cannot dispatch and a request is raised. How fast?",
        target_answer=_ANSWERS[kind],
        question_type=kind,
        facts=group(source(passage_text="An urgent request is answered in 4 hours.")),
    )


@pytest.mark.parametrize(
    ("kind", "gate"),
    [
        (QuestionType.IMPLICATION, "follows"),
        (QuestionType.APPLICATION, "follows"),
    ],
)
def test_a_derived_question_is_sent_to_its_own_gate(kind: str, gate: str) -> None:
    """And never judged on whether the passages state the answer."""
    recording = Recording(answer=True)

    result = build(recording).check(reasoned(kind))

    assert gate in recording.asked
    assert result.status == "accepted"


class TestAnAggregationIsCheckedArithmetically:
    """A total has a right value rather than a likely one.

    So it is computed here and not asked of a model, which agrees with a
    wrong total often enough to matter and which cannot be shown to have
    added anything up.
    """

    PASSAGES = (
        "The northern site employs 40 people.",
        "The southern site employs 25 people.",
    )

    def aggregation(self, answer: str, *passages: str):
        """One aggregation question with the given answer and material.

        Distinct passage ids, or the group holds one passage and the figures
        the total is made of are never both in front of the gate.
        """
        return candidate(
            question_text="How many people do the two sites employ between them?",
            target_answer=answer,
            question_type=QuestionType.AGGREGATION,
            facts=group(
                *(
                    source(fact_id=at, passage_id=at, passage_text=one)
                    for at, one in enumerate(passages or self.PASSAGES, 1)
                )
            ),
        )

    def test_a_right_total_is_accepted_without_a_call(self) -> None:
        """40 and 25 come to 65, and nothing has to be asked whether they do."""
        recording = Recording(answer=False)

        result = build(recording).check(self.aggregation("65"))

        assert result.status == "accepted"
        assert "computes" not in recording.asked, "the model was asked anyway"

    def test_a_wrong_total_is_refused_without_a_call(self) -> None:
        """The failure a model asked to agree produces."""
        recording = Recording(answer=True)

        result = build(recording).check(self.aggregation("68"))

        assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
        assert "computes" not in recording.asked

    def test_two_years_added_together_are_refused(self) -> None:
        """The arithmetic is right and the quantity is not one.

        This corpus asked what year the first description and the prior
        edition "come to together" and answered 4034, which every check it
        faced confirmed.
        """
        recording = Recording(answer=True)

        result = build(recording).check(
            self.aggregation(
                "4034",
                "The method was first described in 2011.",
                "The prior edition carried the version year 2023.",
            )
        )

        assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
        assert "computes" not in recording.asked

    def test_a_total_in_words_still_goes_to_the_model(self) -> None:
        """What the arithmetic cannot read, it does not judge."""
        recording = Recording(answer=True)

        result = build(recording).check(self.aggregation("sixty-five"))

        assert "computes" in recording.asked, "it should have fallen through"
        assert result.status == "accepted"

    def test_a_figure_ending_a_sentence_is_still_a_figure(self) -> None:
        """A four-digit summand followed by a full stop.

        `figures` read a group of up to three digits or a whole number with
        no separator after it, so 1200 in `1200.` matched neither and the
        total was refused as arithmetic over nothing.
        """
        recording = Recording(answer=False)

        result = build(recording).check(
            self.aggregation(
                "1500",
                "The northern site employs 1200.",
                "The southern site employs 300.",
            )
        )

        assert result.status == "accepted"
        assert "computes" not in recording.asked

    def test_a_figure_before_a_comma_is_still_a_figure(self) -> None:
        """The other separator a sentence puts after a number."""
        recording = Recording(answer=False)

        result = build(recording).check(
            self.aggregation(
                "1500",
                "The northern site employs 1200, at two depots.",
                "The southern site employs 300, at one.",
            )
        )

        assert result.status == "accepted"

    def test_two_years_ending_their_sentences_are_still_refused(self) -> None:
        """The 4034 case as the corpus actually writes it.

        Both years were invisible to `figures`, so this was refused for
        finding no arithmetic rather than for summing two calendar points.
        """
        recording = Recording(answer=True)

        result = build(recording).check(
            self.aggregation(
                "4034",
                "The method was first described in 2011.",
                "The prior edition carried the version year 2023.",
            )
        )

        assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
        assert "computes" not in recording.asked


@pytest.mark.parametrize("kind", [QuestionType.IMPLICATION, QuestionType.APPLICATION])
def test_a_conclusion_that_does_not_follow_is_refused(kind: str) -> None:
    """The gate has to be able to say no, or it is not a gate."""
    from database.qa_generator import QuestionRejection

    recording = Recording(answer=False)

    result = build(recording).check(reasoned(kind))

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE


def test_a_retrieval_question_still_faces_recall() -> None:
    """The derived path must not swallow the types it was not built for."""
    recording = Recording(answer=False)

    build(recording).check(candidate(question_type=QuestionType.FACTOID))

    assert "read" in recording.asked
    assert "follows" not in recording.asked


# ── What reaches the row ───────────────────────────────────────────────────


def test_the_level_is_carried_onto_the_question() -> None:
    """Written from the spec, so a reader can group on it without re-deriving."""
    recording = Recording(answer=True)

    result = build(recording).check(reasoned(QuestionType.IMPLICATION))

    assert result.cognitive_level == CognitiveLevel.ANALYSE


def test_a_retrieval_question_carries_its_level_too() -> None:
    """Every type declares one, so every row gets one."""
    recording = Recording()

    result = build(recording).check(candidate(question_type=QuestionType.FACTOID))

    assert result.cognitive_level == CognitiveLevel.RECALL
