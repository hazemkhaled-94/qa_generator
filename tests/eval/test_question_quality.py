"""How a real served model writes and checks questions.

Never a gate, for the same reason `test_extraction_quality.py` is not: a
model's answers move between versions, between quantisations and between runs
at the same temperature, so a threshold here would fail on somebody else's
Tuesday rather than on a regression. It prints its numbers.

The one thing it does assert is that the round-trip gate splits the golden
cases the right way round. That is not a measurement of the model's taste -
it is whether the gate is wired up at all, and a gate that accepts everything
is indistinguishable from no gate.

Needs a model: set LLM_MODEL and LLM_BASE_URL, and QUESTIONS_VERIFIER_MODEL
for the second one, then run

    make test-eval

Skipped otherwise, which is every CI run.
"""

from __future__ import annotations

import os

import pytest
from factories import group, source

from database.qa_generator import QuestionRejection

pytestmark = [pytest.mark.eval, pytest.mark.nlp]

#: Questions whose verdict is not a matter of taste. Each is a question, the
#: passage it cites, and whether the answer is genuinely in that passage.
#:
#: The German case is the one that made the round trip the gate rather than
#: the similarity check it replaced. "Ein Liquiditätsmanagementtool ist eine
#: einjährige Rückgabefrist" survived an NLI model, an LLM judge and a
#: structural check, because it reads like its passage. It does not survive
#: being asked.
GOLDEN = (
    {
        "name": "answered outright",
        "question": "Within how many hours is a standard support request answered?",
        "passage": (
            "Standard requests are answered within 48 hours on working days. "
            "Urgent requests are answered within 4 hours."
        ),
        "target": "48 hours",
        "language": "en",
        "recoverable": True,
    },
    {
        "name": "answered in other words",
        "question": "How quickly does an urgent support request get a reply?",
        "passage": (
            "Standard requests are answered within 48 hours on working days. "
            "Urgent requests are answered within 4 hours."
        ),
        "target": "4 hours",
        "language": "en",
        "recoverable": True,
    },
    {
        "name": "the LMT case: reads like the passage, is not in it",
        "question": "Was ist ein Liquiditätsmanagementtool?",
        "passage": (
            "Für den Fonds gilt eine einjährige Rückgabefrist. "
            "Die Verwaltungsgesellschaft veröffentlicht die Frist im "
            "Verkaufsprospekt."
        ),
        "target": "eine einjährige Rückgabefrist",
        "language": "de",
        "recoverable": False,
    },
    {
        "name": "a plausible question the passage does not cover",
        "question": (
            "Within how many hours is a standard support request answered "
            "on a public holiday?"
        ),
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": None,
        "language": "en",
        "recoverable": False,
    },
    {
        "name": "the right subject, the wrong number",
        "question": "Within how many hours is a standard support request answered?",
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": "4 hours",
        "language": "en",
        "recoverable": False,
    },
)


@pytest.fixture(scope="module")
def checker():
    """The real gates against the served models, or a skip."""
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is unset; no model to measure")

    from dataclasses import replace

    from llm.client import Client, ModelUnavailable
    from llm.config import Settings
    from question_generation.config import Settings as QuestionSettings
    from question_generation.embedding import Embedder
    from question_generation.verification import QuestionChecker, Verifier

    settings = Settings.load()
    questions = QuestionSettings.load()
    if not questions.verifier_model:
        print("\nQUESTIONS_VERIFIER_MODEL is unset; the writer is marking its own work")
    verifier = Verifier(
        Client(
            replace(settings, model=questions.verifier_model)
            if questions.verifier_model
            else settings
        )
    )
    try:
        verifier.recover("Is this on?", ["This is on."])
    except ModelUnavailable as exc:
        pytest.skip(f"the verifier is not answering: {exc}")

    return QuestionChecker(
        embedder=Embedder(questions.embedding_model, questions.max_tokens),
        verifier=verifier,
        # No probe: these cases are about recoverability, and a database the
        # eval layer does not have would only add a way for them to fail.
        nearest=lambda embedding: None,
        threshold=questions.duplicate_cosine,
    )


def candidate_for(example: dict):
    """Builds the candidate one golden case describes."""
    from question_generation.models import Candidate

    return Candidate(
        question_text=example["question"],
        target_answer=example["target"],
        answerable=example["target"] is not None,
        group=group(
            source(
                1,
                statement=example["passage"].split(". ")[0] + ".",
                language=example["language"],
                passage_text=example["passage"],
            )
        ),
    )


@pytest.mark.parametrize("example", GOLDEN, ids=lambda one: one["name"])
def test_the_gate_splits_the_golden_cases(checker, example) -> None:
    """Recoverable cases pass; the rest are stopped, whichever gate does it."""
    result = checker.check(candidate_for(example))

    print(f"\n{example['name']}: {result.status} {result.rejected_reason or ''}")

    if example["recoverable"]:
        assert result.accepted, (
            f"a recoverable question was rejected: {result.rejected_reason}"
        )
    else:
        assert not result.accepted, (
            "a question the passage does not answer was accepted"
        )
        assert result.rejected_reason in (
            QuestionRejection.NOT_RECOVERABLE,
            QuestionRejection.ANSWERABLE_AFTER_ALL,
        ), result.rejected_reason


def test_the_whole_golden_set_is_scored(checker) -> None:
    """One number for the set, printed rather than asserted against."""
    right = 0
    for example in GOLDEN:
        result = checker.check(candidate_for(example))
        right += result.accepted == example["recoverable"]

    print(f"\ngolden set: {right} of {len(GOLDEN)} judged as expected")

    assert right, "the gate agreed with none of the golden cases"


def test_the_writer_produces_a_usable_question_from_a_fact(checker) -> None:
    """Measured, not asserted: what it writes is the model's taste."""
    from llm.client import Client
    from llm.config import Settings
    from question_generation.generation import QuestionWriter

    written = QuestionWriter(Client(Settings.load())).write(
        group(
            source(
                1,
                statement="A standard support request is answered within 48 hours.",
                language="en",
            )
        ),
        answerable=True,
    )

    print(f"\nwritten: {written.question_text!r} -> {written.target_answer!r}")

    assert written.question_text, "the model wrote nothing"
