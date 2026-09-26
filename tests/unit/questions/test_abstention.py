"""A judgement that could not be had is recorded, not read as a pass.

The checker rejects a question when a judgement comes back False. An
abstention comes back None, and `None is not False` - so a gate whose model
was unreachable read exactly like a gate that passed, and the question was
ACCEPTED on a judgement nobody made. Nothing on the row said so, which made
it invisible in the released set and impossible to find afterwards.

Abstaining is still right: losing one opinion about wording must not throw
away a question the passages support. What is under test here is that the
abstention leaves a trace.
"""

from __future__ import annotations

import pytest
from factories import candidate

from question_generation.checker import QuestionChecker

pytestmark = pytest.mark.nlp


class Judge:
    """A phrasing judge whose answer is the same for every judgement."""

    def __init__(self, answer: bool | None) -> None:
        """Initialises with what it will say, None being an abstention."""
        self.answer = answer
        self.asked: list[str] = []

    def names_its_source(self, question: str) -> bool | None:
        """Whether the question says where its answer is."""
        self.asked.append("names_its_source")
        return self.answer

    def names_something(self, question: str) -> bool | None:
        """Whether it names anything a searcher could have typed."""
        self.asked.append("names_something")
        return self.answer

    def self_contained(self, question: str, pointers) -> bool | None:
        """Whether its pointing words land inside it."""
        self.asked.append("self_contained")
        return self.answer


def build(judge: Judge) -> QuestionChecker:
    """A checker whose only opinion is the one handed over."""
    return QuestionChecker(
        embedder=None,
        verifier=None,
        nearest=lambda embedding: None,
        threshold=0.93,
        bounds={"value": (1, 80), "list": (3, 300), "explanation": (20, 600)},
        phrasing=judge,
    )


def test_an_abstained_judgement_is_recorded() -> None:
    """The defect: it used to leave nothing at all behind."""
    judge = Judge(None)
    abstained: list[str] = []

    failed = build(judge)._phrasing(candidate(), abstained)

    assert failed is None, "an abstention must not reject a question"
    assert judge.asked, "the judge was never asked, so this proves nothing"
    assert abstained == judge.asked


def test_a_judgement_that_answered_records_nothing() -> None:
    """Only what could not be had, so an empty list means every gate ran."""
    judge = Judge(True)
    abstained: list[str] = []

    build(judge)._phrasing(candidate(), abstained)

    assert judge.asked
    assert abstained == []


def test_an_abstention_is_not_a_rejection() -> None:
    """None is not an answer, and the whole gate turns on the difference.

    The default candidate reaches one judgement, `names_its_source`, where
    True is the refusing answer: a question that says where its answer is
    has answered half of itself. So a judge that could not be reached must
    leave the verdict where the ACQUITTING answer leaves it, and must not
    be read as either.
    """
    refused = build(Judge(True))._phrasing(candidate(), [])
    acquitted = build(Judge(False))._phrasing(candidate(), [])
    abstained = build(Judge(None))._phrasing(candidate(), [])

    assert refused is not None
    assert acquitted is None
    assert abstained is None


def test_the_verdict_carries_what_abstained() -> None:
    """So the row outlives the run that could not get the judgement."""
    one = candidate()

    verdict = QuestionChecker._verdict(one, None, None, abstained=("names_its_source",))

    assert verdict.gates_abstained == ("names_its_source",)
    assert verdict.accepted, "recording an abstention must not change the verdict"


def test_a_verdict_with_no_judge_records_none() -> None:
    """The default, which the re-check in `service.py` relies on."""
    assert QuestionChecker._verdict(candidate(), None, None).gates_abstained == ()
