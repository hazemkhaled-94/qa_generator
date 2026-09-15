"""Every gate, and the order they are applied in.

The order is not a detail. Each gate that fires saves the cost of the ones
behind it, and the only expensive one is last, so a test that lets a
malformed question reach the verifier is a test of a bill.
"""

from __future__ import annotations

import pytest
from factories import candidate, group, source

from database.qa_generator import QuestionRejection, QuestionStatus
from question_generation.models import Neighbour
from question_generation.verification import (
    QuestionChecker,
    Reading,
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
        self,
        vector: list[float] | None = None,
        recovers: str | None = "4 kg",
        stands_alone: bool = True,
    ):
        """Initialises with what to answer, and nothing asked yet."""
        self.vector = vector or [1.0] + [0.0] * 1023
        self.recovers = recovers
        self.stands_alone = stands_alone
        self.embedded = 0
        self.verified = 0

    def embed(self, text: str) -> list[float]:
        """Answers with the one vector, and counts the ask."""
        self.embedded += 1
        return self.vector

    def read(self, question: str, passages) -> Reading:
        """Answers with the scripted reading, and counts the ask."""
        self.verified += 1
        return Reading(recovered=self.recovers, stands_alone=self.stands_alone)


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

        def read(self, question: str, passages) -> Reading:
            """Keeps the passages and answers as scripted."""
            seen.append(list(passages))
            return super().read(question, passages)

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


# ── The gates that judge it as a question ──────────────────────────────────
#
# Every example below is a row this stage actually wrote, before the prompt
# was given the passage and these gates were added.


@pytest.mark.parametrize(
    ("question", "fact"),
    [
        (
            "Was schüren geopolitische Konflikte?",
            "Geopolitische Konflikte schüren Unsicherheit.",
        ),
        (
            "Was umfasst Kreditgeschäfte?",
            "Kreditgeschäfte umfassen Bilanzaktiva und außerbilanzielle Geschäfte.",
        ),
        (
            "Was prägt das Umfeld des Finanzsektors?",
            "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen.",
        ),
    ],
)
def test_a_question_that_only_permutes_its_fact_is_refused(question, fact) -> None:
    """The defect this stage shipped with, one real row at a time.

    Each is the fact with one part replaced by a question word. Nobody
    searching a corpus of thousands of pages types any of them.
    """
    failed = checked(
        question_text=question,
        target_answer="Unsicherheit",
        language="de",
        statements=(fact,),
    )

    assert code(failed) == QuestionRejection.RESTATES_FACT


def test_a_question_naming_something_its_fact_does_not_is_kept() -> None:
    """The gate has to let a real question through or it is just a filter.

    `Bundesbank` and `Vorschlag` are in the passage and not in the fact, so
    the question carries something a searcher would have had to know.
    """
    failed = checked(
        question_text=(
            "Welchen Vorschlag haben Bafin und Bundesbank für kleine Banken gemacht?"
        ),
        target_answer="eine Bilanzsumme unter 10 Milliarden Euro",
        language="de",
        statements=("Die kleinen Banken müssen eine Bilanzsumme haben.",),
    )

    assert failed is None


def test_a_question_sharing_its_subject_with_its_fact_is_not_a_restatement() -> None:
    """Reusing the subject's words is how a question is about the subject.

    Only adding *nothing at all* is the failure, so a question that names
    the document its rule comes from passes on that word alone.
    """
    failed = checked(
        question_text="Wie oft verlangt die MaRisk einen Risikobericht?",
        target_answer="vierteljährlich",
        language="de",
        statements=("Ein Risikobericht muss vierteljährlich erstellt werden.",),
    )

    assert failed is None


def test_the_restatement_gate_leaves_unanswerable_questions_alone() -> None:
    """A perturbation is meant to depart from its fact, not to restate it.

    It is written from the fact and deliberately not answered by it, so
    comparing the two for overlap says nothing about whether it is good.
    """
    failed = checked(
        question_text="Was schüren geopolitische Konflikte im Jahr 2030?",
        target_answer=None,
        answerable=False,
        language="de",
        statements=("Geopolitische Konflikte schüren Unsicherheit.",),
    )

    assert failed is None


@pytest.mark.parametrize(
    "answer",
    [
        "Verwarnungen aussprechen",
        "nachvollziehbar zu begründen",
        "nahmen Produkte vom Markt",
    ],
)
def test_an_answer_that_describes_an_action_is_refused(answer) -> None:
    """Three real target answers, none of them a thing anyone asked for.

    Any verb and not only a finite one: two of these are infinitives, so
    they make no claim and are still not answers.
    """
    failed = checked(
        question_text="Welche Maßnahme hat die Bafin 2025 ergriffen?",
        target_answer=answer,
        language="de",
        statements=("Die Bafin sprach Verwarnungen aus.",),
    )

    assert code(failed) == QuestionRejection.MALFORMED


@pytest.mark.parametrize(
    "answer",
    ["48 Stunden", "2025", "1.033", "der Vorstand", "vierteljährlich", "Hamburg"],
)
def test_an_answer_that_names_a_thing_is_kept(answer) -> None:
    """A value, a date, an amount, a name, a period, a place."""
    failed = checked(
        question_text="Welchen Wert nennt die Gebührenverordnung der Bafin?",
        target_answer=answer,
        language="de",
        statements=("Die Verordnung nennt einen Wert.",),
    )

    assert failed is None


def test_a_question_asking_two_things_is_refused() -> None:
    """A chatbot answering one of them is neither right nor wrong."""
    failed = checked(
        question_text="Wie hoch ist die Gebühr? Und wer erhebt sie?",
        target_answer="1.033 Euro",
        language="de",
    )

    assert code(failed) == QuestionRejection.MALFORMED


def test_a_question_nobody_could_have_asked_cold_is_refused() -> None:
    """The verifier's judgement, on the call it was already making.

    A question can be perfectly answerable by the passages it cites and
    still be one nobody would type, which no structural check can see.
    """
    recording = Recording(recovers="4 kg", stands_alone=False)

    result = build(recording).check(candidate())

    assert result.rejected_reason == QuestionRejection.UNANCHORED
    assert recording.verified == 1, "it must not cost a second call"


def test_an_unanchored_unanswerable_question_is_refused_too() -> None:
    """A chatbot declining a question nobody would ask proves nothing."""
    recording = Recording(recovers=None, stands_alone=False)

    result = build(recording).check(
        candidate(
            question_text="Wie schwer ist es auf dem Mars?",
            target_answer=None,
            answerable=False,
        )
    )

    assert result.rejected_reason == QuestionRejection.UNANCHORED


def test_the_phrasing_judgement_is_made_before_the_answer_is() -> None:
    """A question nobody would ask is refused whatever its answer does.

    Reported as `unanchored` rather than `not_recoverable`, because those
    say different things about what to change.
    """
    recording = Recording(recovers=None, stands_alone=False)

    result = build(recording).check(candidate())

    assert result.rejected_reason == QuestionRejection.UNANCHORED


def test_a_writer_marking_its_own_work_cannot_reject_on_phrasing() -> None:
    """The only gate here that is an opinion, and opinions need a holder.

    A 4B model judging its own questions rejected `According to the ECB and
    NCAs, who conducts the due diligence check for an outsourcing
    arrangement?` for naming nothing. A gate losing questions that good is
    worse than no gate, so without an independent verifier the verdict is
    logged and the question kept.
    """
    recording = Recording(recovers="4 kg", stands_alone=False)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        judge_phrasing=False,
    )

    result = checker.check(candidate())

    assert result.accepted, "a self-judged opinion rejected a question"


def test_recoverability_is_judged_even_when_phrasing_is_not() -> None:
    """One is an opinion; the other is checkable against the passage.

    Turning the subjective gate off must not turn off the gate that reads
    the evidence, which is the one that matters most.
    """
    recording = Recording(recovers=None, stands_alone=False)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        judge_phrasing=False,
    )

    result = checker.check(candidate())

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE


def test_the_free_gates_are_unaffected_by_who_is_verifying() -> None:
    """A measurement does not need a second opinion to be worth making."""
    recording = Recording()
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        judge_phrasing=False,
    )

    result = checker.check(
        candidate(
            question_text="Was schüren geopolitische Konflikte?",
            target_answer="Unsicherheit",
            facts=group(
                source(
                    1,
                    statement="Geopolitische Konflikte schüren Unsicherheit.",
                    language="de",
                ),
            ),
        )
    )

    assert result.rejected_reason == QuestionRejection.RESTATES_FACT
