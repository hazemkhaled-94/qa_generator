"""Every gate, and the order they are applied in.

The order is not a detail. Each gate that fires saves the cost of the ones
behind it, and the only expensive one is last, so a test that lets a
malformed question reach the verifier is a test of a bill.
"""

from __future__ import annotations

import pytest
from factories import candidate, group, source

from database.qa_generator import (
    AnswerForm,
    QuestionRejection,
    QuestionStatus,
    QuestionType,
)
from question_generation.checker import QuestionChecker
from question_generation.gates import (
    OVERLAP,
    agrees,
    anchored,
    asks_for_an_agent,
    compares,
    enumerates,
    explains,
    fitting,
    incomplete,
    moves_on,
    names_parties,
    near_verdict,
    on_topic,
    periods,
    restates,
    same_material,
    structural,
)
from question_generation.models import Neighbour
from question_generation.verifier import Reading

pytestmark = pytest.mark.nlp


def code(failed) -> str | None:
    """The gate's code out of a verdict, or None when it passed."""
    return failed[0] if failed else None


#: What each form is held to unless a test says otherwise. The value floor is
#: 1, as the setting ships: a floor of 15 refused 41% of the answers one
#: corpus had accepted.
BOUNDS = {"value": (1, 80), "list": (3, 300), "explanation": (20, 600)}


def checked(**kwargs):
    """Runs the free gates over one candidate's fields.

    Takes `form` singular and hands back the verdict alone. `structural`
    takes every form a type will accept and reports which one the answer
    turned out to fit; these tests are about the gates rather than about
    that choice, so they pin one form and read the verdict. What happens
    when a type allows several is covered in `test_fitting` below.
    """
    fields = {
        "question_text": "What does the device weigh?",
        "target_answer": "4 kg",
        "answerable": True,
        "language": "en",
        "statements": (),
        "form": AnswerForm.VALUE,
        "bounds": BOUNDS,
        "titles": (),
        **kwargs,
    }
    failed, _ = structural(forms=(fields.pop("form"),), **fields)
    return failed


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


# Every pair below is one the verifier really produced. The four marked SAME
# were all refused by the string containment this replaced, which is what
# made `not_recoverable` the largest bucket of rejections.


@pytest.mark.parametrize(
    ("recovered", "target"),
    [
        (
            "Die Darstellung der zukünftigen Entwicklung und der Risiken.",
            "zukünftige Entwicklung und Risiken",
        ),
        (
            (
                "kapitalbildende Lebensversicherungen, die einen angemessenen "
                "Kundennutzen aufweisen."
            ),
            "kapitalbildend und einen angemessenen Kundennutzen",
        ),
        (
            "Digitalisierung, Nachhaltigkeit, geopolitische Umbrüche",
            "Digitalisierung und Nachhaltigkeit",
        ),
    ],
)
def test_an_inflection_is_not_a_disagreement(recovered, target) -> None:
    """German declines, and a declension is not a different answer.

    Compared on content lemmas for that reason. Containment refused all
    three, and a person reading either column calls them the same.
    """
    assert agrees(recovered, target, "de")


@pytest.mark.parametrize(
    ("recovered", "target"),
    [
        ("die qualitative Aufsicht in Deutschland", "Regelungsrahmen"),
        (
            "Konditionen- und Strukturbeitrag im Zinsbuch",
            "Abgrenzung der Erfolgsquellen",
        ),
        (
            "die Art der vergebenen Kredite und die Kreditvergabepraxis der Fonds",
            "langer Anlagehorizont",
        ),
        (
            (
                "einfacheren Verschuldungsquote und deutlich über dem "
                "Basel-III-Mindestwert von 3 Prozent"
            ),
            "kleinere und nicht-komplexe Banken",
        ),
    ],
)
def test_a_different_answer_is_still_refused(recovered, target) -> None:
    """Loosening the comparison must not accept a wrong answer.

    These are the pairs where the verifier answered something else, and none
    of them may pass: a question scored against an answer the material does
    not give marks a chatbot wrong for being right.
    """
    assert not agrees(recovered, target, "de")


def test_an_answer_claiming_more_than_was_found_is_refused() -> None:
    """Five things asked for and three recovered is not agreement."""
    assert not agrees(
        "Genauigkeit, Stabilität und Konsistenz der Verfahren",
        "die Qualität der Modellergebnisse, die Genauigkeit, die Stabilität, "
        "die Konsistenz und die Erklärbarkeit",
        "de",
    )


def test_a_lemma_the_pipeline_gets_wrong_is_a_known_miss() -> None:
    """`de_core_news_md` leaves `Umbrüchen` alone and lemmatises `Umbrüche`.

    So these two, which are the same answer, do not meet. Recorded rather
    than fixed: stemming would close it and would open false accepts, and
    twelve pairs is not enough evidence to take that trade.
    """
    assert not agrees(
        "geopolitische Umbrüchen und fortschreitender Digitalisierung",
        "geopolitische Umbrüche und fortschreitende Digitalisierung",
        "de",
    )


def test_a_target_with_no_content_word_falls_back_to_containment() -> None:
    """A bare yes has no lemma to compare, so containment is all there is."""
    assert agrees("ja, das ist zulässig", "ja", "de")
    assert not agrees("nein", "ja", "de")


# ── The gates in order ─────────────────────────────────────────────────────


class Recording:
    """An embedder and a verifier that count how often they were asked."""

    def __init__(
        self,
        vector: list[float] | None = None,
        recovers: str | None = "4 kg",
        stands_alone: bool = True,
        names_its_source: bool = False,
        backs: bool = False,
        self_contained: bool = True,
    ):
        """Initialises with what to answer, and nothing asked yet."""
        self.vector = vector or [1.0] + [0.0] * 1023
        self.recovers = recovers
        self.stands_alone = stands_alone
        #: What the phrasing judge answers, for the residue a rule leaves.
        #: Named apart from the methods that serve them, which the checker
        #: calls and which count the ask.
        self.names_source = names_its_source
        self.contained = self_contained
        self.phrased = 0
        #: What the entailment pass answers. False by default, so a test
        #: about recall measures recall: the pass can only ever rescue, and
        #: one that always said yes would hide every rejection below it.
        self.backs = backs
        self.threaded: tuple = ()
        self.asked: str = ""
        self.embedded = 0
        self.verified = 0
        self.supported = 0

    def embed(self, text: str) -> list[float]:
        """Answers with the one vector, and counts the ask."""
        self.embedded += 1
        return self.vector

    def read(self, question: str, passages, thread=()) -> Reading:
        """Answers with the scripted reading, and counts the ask."""
        self.verified += 1
        self.threaded = tuple(thread)
        return Reading(recovered=self.recovers)

    def names_its_source(self, question: str) -> bool:
        """The residue judgement, for a question no rule settled."""
        self.phrased += 1
        return self.names_source

    def names_something(self, question: str) -> bool:
        """Whether a question the parse called thin names anything."""
        self.phrased += 1
        return self.stands_alone

    def self_contained(self, question: str, pointers) -> bool:
        """Whether the pointers a parse found land inside the question."""
        self.phrased += 1
        return self.contained

    def supports(self, question: str, answer: str, passages, thread=()) -> bool:
        """Answers the entailment pass, and counts the ask."""
        self.supported += 1
        return self.backs

    def computes(self, question: str, answer: str, passages, thread=()) -> bool:
        """Answers the derived-answer pass, and counts the ask."""
        self.supported += 1
        return self.backs


def build(recording: Recording, near: Neighbour | None = None) -> QuestionChecker:
    """A checker over a scripted probe and a scripted pair of models."""
    return QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: near,
        threshold=0.93,
        phrasing=recording,
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

        def read(self, question: str, passages, thread=()) -> Reading:
            """Keeps the passages and answers as scripted."""
            seen.append(list(passages))
            return super().read(question, passages, thread)

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

    assert result.criteria.document_scope == "cross_document"
    assert result.fact_ids == (1, 2)


# ── The gates that judge it as a question ──────────────────────────────────
#
# Every example below is a row this stage actually wrote, before the prompt
# was given the passage and these gates were added.


# ── Why there is no gate for "the question is its own fact" ───────────────
#
# One was written and removed. These keep the finding, because the idea is an
# appealing one and the next person to have it should be able to see what it
# costs before writing it again.


@pytest.mark.parametrize(
    ("question", "answer", "fact"),
    [
        (
            "Wie hoch war die Arbeitslosenquote im August 2025?",
            "6,4 Prozent",
            "Die Arbeitslosenquote liegt im August 2025 bei 6,4 Prozent.",
        ),
        (
            "Wie viele Banken in Deutschland könnten sich qualifizieren?",
            "Etwa 1.000 Banken",
            "Etwa 1.000 Banken in Deutschland könnten sich qualifizieren.",
        ),
        (
            "Wie oft muss die Angemessenheit der Verfahren überprüft werden?",
            "jährlich",
            "Die Angemessenheit der Verfahren ist zumindest jährlich zu prüfen.",
        ),
    ],
)
def test_a_good_question_is_its_fact_minus_the_answer(question, answer, fact) -> None:
    """Which is why no lexical gate can refuse one for being that.

    For a single atomic fact, asking about it IS reproducing it without its
    answer. Every question here does exactly that and every one is as good
    as a benchmark question gets; a gate refusing the shape refused all
    three, measured on 61 real rows.
    """
    assert (
        checked(
            question_text=question,
            target_answer=answer,
            language="de",
            statements=(fact,),
        )
        is None
    )


def test_a_vague_question_has_the_same_shape_and_is_not_refused_here() -> None:
    """The one the removed gate was written for, kept to show the problem.

    It has the same overlap with its fact as the three above. What makes it
    worse is that its answer is not determinate - and that is not lexical
    either: requiring a number or a name in the answer refused 7 of 15
    accepted answers. So the verifier judges it, under
    `not_recoverable`, where it can be judged.
    """
    assert (
        checked(
            question_text="Was schüren geopolitische Konflikte?",
            target_answer="Unsicherheit",
            language="de",
            statements=("Geopolitische Konflikte schüren Unsicherheit.",),
        )
        is None
    )


def test_a_fact_handed_back_with_a_question_mark_is_still_refused() -> None:
    """The exact-match check stays: that one is not a judgement call."""
    failed = checked(
        question_text="Geopolitische Konflikte schüren Unsicherheit?",
        target_answer="Unsicherheit",
        language="de",
        statements=("Geopolitische Konflikte schüren Unsicherheit.",),
    )

    assert code(failed) == QuestionRejection.MALFORMED


@pytest.mark.parametrize(
    "answer",
    [
        "Verwarnungen aussprechen",
        "nachvollziehbar zu begründen",
        "nahmen Produkte vom Markt",
    ],
)
def test_an_answer_that_describes_an_action_is_refused_for_a_value(answer) -> None:
    """Three real target answers, none of them a thing anyone asked for.

    Any verb and not only a finite one: two of these are infinitives, so
    they make no claim and are still not answers.

    Only where a value was asked for. The same rule applied to every answer
    is what made `why` and `how` unwritable: their answers are supposed to
    carry a verb.
    """
    failed = checked(
        question_text="Welche Maßnahme hat die Bafin 2025 ergriffen?",
        target_answer=answer,
        language="de",
        form=AnswerForm.VALUE,
        statements=("Die Bafin sprach Verwarnungen aus.",),
    )

    assert code(failed) == QuestionRejection.WRONG_FORM


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


def test_two_sentences_each_ending_in_a_question_mark_are_malformed() -> None:
    """The shape a question mark counts. `compound` catches the other one."""
    failed = checked(
        question_text="Wie hoch ist die Gebühr? Und wer erhebt sie?",
        target_answer="1.033 Euro",
        language="de",
    )

    assert code(failed) == QuestionRejection.MALFORMED


def test_a_question_nobody_could_have_asked_cold_is_refused() -> None:
    """Two holders agreeing: the parse calls it thin and the judge agrees.

    A question can be perfectly answerable by the passages it cites and
    still be one nobody would type. The parse only says the question is thin
    enough for the judgement to be worth asking for.
    """
    recording = Recording(recovers="4 kg", stands_alone=False)

    result = build(recording).check(
        candidate(question_text="What specific components are included?")
    )

    assert result.rejected_reason == QuestionRejection.UNANCHORED
    assert recording.verified == 0, "the passages are never read for this"


def test_an_unanchored_unanswerable_question_is_refused_too() -> None:
    """A chatbot declining a question nobody would ask proves nothing."""
    recording = Recording(recovers=None, stands_alone=False)

    result = build(recording).check(
        candidate(
            question_text="What specific components are included?",
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

    result = build(recording).check(
        candidate(question_text="What specific components are included?")
    )

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
            question_text="Geopolitische Konflikte schüren Unsicherheit?",
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

    assert result.rejected_reason == QuestionRejection.MALFORMED


# ── The answer floor, and the criteria the gates read off ──────────────────


@pytest.mark.parametrize("answer", ["7", "8%", "Nein", "70%", "2025"])
def test_an_answer_under_the_floor_is_refused(answer) -> None:
    """A blunt measure, chosen deliberately.

    At fifteen characters it refuses `70%` and `2025` as well as `7` and
    `Nein` - measured on one corpus, 41% of the answers that had been
    accepted. QUESTIONS_MIN_ANSWER_CHARS is where that is decided.
    """
    failed = checked(target_answer=answer, bounds={**BOUNDS, "value": (15, 80)})

    assert code(failed) == QuestionRejection.ANSWER_TOO_SHORT


@pytest.mark.parametrize(
    "answer", ["European Supervisory Authorities", "29 January 2027", "Article 4(1)(1)"]
)
def test_an_answer_over_the_floor_is_kept(answer) -> None:
    """A name, a date and a citation, all of them scoreable."""
    assert checked(target_answer=answer, bounds={**BOUNDS, "value": (15, 80)}) is None


def test_a_floor_of_zero_refuses_nothing_for_length() -> None:
    """Which is what keeps the bare values a corpus of numbers needs."""
    assert checked(target_answer="70%", bounds=BOUNDS) is None


def test_an_unanswerable_question_is_never_too_short() -> None:
    """It has no answer to measure, and is scored on behaviour."""
    assert (
        checked(
            answerable=False, target_answer=None, bounds={**BOUNDS, "value": (15, 80)}
        )
        is None
    )


def test_a_follow_up_is_not_judged_on_standing_alone() -> None:
    """`And for an urgent one?` names nothing, and is the point.

    Leaning on the thread is what makes a follow-up a follow-up, so judging
    one as though it had been asked cold would reject every one of them.
    """
    recording = Recording(recovers="4 hours", stands_alone=False)

    result = build(recording).check(
        candidate(
            question_text="And for an urgent one?",
            target_answer="4 hours",
            thread=(("How long for a standard one?", "48 hours"),),
        )
    )

    assert result.accepted
    assert result.thread_position == 2


def test_the_verifier_is_shown_the_thread_when_it_reads_a_follow_up() -> None:
    """Read alone, a follow-up has no answer in any passage."""
    recording = Recording(recovers="4 kg")

    build(recording).check(
        candidate(
            question_text="And for an urgent one?",
            target_answer="4 hours",
            thread=(("How long for a standard one?", "48 hours"),),
        )
    )

    assert recording.threaded == (("How long for a standard one?", "48 hours"),)


def test_a_root_question_is_still_judged_on_standing_alone() -> None:
    """The exemption is for follow-ups and for nothing else."""
    recording = Recording(recovers="4 kg", stands_alone=False)

    result = build(recording).check(
        candidate(question_text="What specific components are included?")
    )

    assert result.rejected_reason == QuestionRejection.UNANCHORED


def test_the_three_criteria_are_read_off_the_facts_the_question_cites() -> None:
    """Each says something different about what a chatbot has to do."""
    across = group(
        source(1, document="a", passage_id=1, topic_id=7),
        source(2, document="b", passage_id=2, topic_id=8),
    )

    result = build(Recording()).check(candidate(facts=across))

    assert result.criteria.passage_scope == "multi_passage"
    assert result.criteria.document_scope == "cross_document"
    assert result.criteria.topic_scope == "multi_topic"
    assert result.criteria.difficulty == "hard", "three of the five"
    assert result.criteria.score == 3


def test_a_follow_up_counts_towards_its_own_difficulty() -> None:
    """Carrying a thread is one of the things that makes answering harder.

    Measured on the score rather than on the band: one point does not
    always cross a boundary, and the score is what the band is read from.
    """
    alone = build(Recording()).check(candidate())
    followed = build(Recording()).check(
        candidate(thread=(("How long for a standard one?", "48 hours"),))
    )

    assert followed.criteria.score == alone.criteria.score + 1
    assert followed.criteria.follows and not alone.criteria.follows


# ── Naming the source, which is the one thing a question must not do ───────
#
# Every question below is a row this stage really wrote, under a prompt that
# asked it to name what it was asking about. Seven of ten in one topic named
# the document instead.


@pytest.mark.parametrize(
    "question",
    [
        "Laut den 'Risiken im Fokus 2026', wie viele Risiken werden genannt?",
        "Welche Themen werden in den 'Risiken im Fokus 2026' beschrieben?",
        "In Risiken im Fokus 2026, welcher Anteil entfiel auf Phishing?",
    ],
)
def test_a_question_quoting_its_own_document_title_is_refused(question) -> None:
    """The free half of the gate, and it runs before any model call.

    The titles are rows in this corpus's documents table, and the writer
    echoed them straight back.
    """
    failed = checked(
        question_text=question,
        target_answer="drei",
        language="de",
        titles=("Risiken im Fokus 2026",),
    )

    assert code(failed) == QuestionRejection.LEAKS_SOURCE


def test_a_title_too_short_to_be_a_name_is_not_matched() -> None:
    """`Contents` and `March 2018` are titles in this corpus.

    A question may contain either by accident, so matching on them would
    refuse questions that cite nothing.
    """
    assert (
        checked(
            question_text="What are the contents of a risk report?",
            target_answer="the exposures and the limits",
            titles=("Contents", "March 2018"),
        )
        is None
    )


def test_a_question_naming_no_title_passes_the_free_half() -> None:
    """Naming a party or a period is not naming a source."""
    assert (
        checked(
            question_text="Wie viele Cybervorfälle wurden der Bafin 2025 gemeldet?",
            target_answer="dreihundert",
            language="de",
            titles=("Risiken im Fokus 2026", "Druckversion - Jahresbericht 2025"),
        )
        is None
    )


def test_the_verifier_catches_a_source_it_named_without_quoting() -> None:
    """`Laut dem Jahresbericht 2025` is not `Druckversion - Jahresbericht 2025`.

    The free half matches a title verbatim, so a paraphrase gets past it.
    `cites_source` catches the patterns, and the model is the residue.
    """
    recording = Recording(recovers="4 kg", names_its_source=True)

    result = build(recording).check(candidate())

    assert result.rejected_reason == QuestionRejection.LEAKS_SOURCE
    assert recording.verified == 0, "the passages are never read for this"


def test_a_follow_up_may_not_name_its_source_either() -> None:
    """Leaning on the conversation is allowed; naming the file is not.

    The two exemptions are different: a follow-up is excused from naming a
    subject, never from hiding its source.
    """
    recording = Recording(recovers="4 hours", names_its_source=True)

    result = build(recording).check(
        candidate(
            question_text="And in the annual report, what about urgent ones?",
            target_answer="4 hours",
            thread=(("How long for a standard one?", "48 hours"),),
        )
    )

    assert result.rejected_reason == QuestionRejection.LEAKS_SOURCE


def test_naming_a_subject_and_naming_a_source_are_judged_separately() -> None:
    """The conflation this gate exists to undo.

    A question that names its subject and no source is what is wanted, and
    for a while the phrasing gate asked for the opposite.
    """
    recording = Recording(recovers="4 kg", stands_alone=True, names_its_source=False)

    result = build(recording).check(candidate())

    assert result.accepted


def test_a_self_judged_verifier_cannot_reject_for_naming_a_source() -> None:
    """The same rule as the phrasing gate: an opinion needs a holder."""
    recording = Recording(recovers="4 kg", names_its_source=True)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        judge_phrasing=False,
    )

    result = checker.check(candidate())

    assert result.accepted


def test_the_free_half_of_the_gate_runs_without_a_verifier_at_all() -> None:
    """A measurement needs no second opinion.

    Quoting a title is not a matter of taste, so it is refused whoever is
    verifying - or whether anything is.
    """
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
            question_text="Laut den 'Risiken im Fokus 2026', wie viele Risiken?",
            facts=group(
                source(
                    1,
                    statement="Es werden drei Risiken genannt.",
                    language="de",
                    document_title="Risiken im Fokus 2026",
                )
            ),
        )
    )

    assert result.rejected_reason == QuestionRejection.LEAKS_SOURCE
    assert recording.verified == 0, "it must not need a call"


@pytest.mark.parametrize("answer", ["dreihundert", "erstmals", "angespannt"])
def test_a_one_word_answer_is_never_read_as_an_action(answer) -> None:
    """A single word is a thing, whatever the tagger calls it.

    `de_core_news_md` tags `dreihundert` as a verb, so checking a one-word
    answer refuses a good one to catch a rare bad one. Every real failure
    this gate was written for was several words long.
    """
    assert checked(target_answer=answer, language="de") is None


def test_a_shared_number_is_not_agreement_on_its_own() -> None:
    """`4 kg` answered a question whose target was `4 hours`, and passed.

    The units check ran first and returned on its own, so a recovered answer
    sharing one number with the target was accepted whatever else it said.
    Both halves have to pass now.
    """
    assert not agrees("4 kg", "4 hours", "en")
    assert agrees("4 hours", "4 hours", "en")


@pytest.mark.parametrize(
    ("recovered", "target"),
    [("die Gebühr ist 1,033 Euro", "1.033 Euro"), ("0,3 Prozent", "0.3 Prozent")],
)
def test_a_number_is_the_same_number_in_either_locale(recovered, target) -> None:
    """This corpus writes `0,3` and the verifier answers `0.3` as often.

    Folded only when comparing one number against another, so `4` is still
    not `48`.
    """
    assert agrees(recovered, target, "de")


# ── The form each type's answer takes, which decides who reads it how ──────


@pytest.mark.parametrize(
    "answer",
    [
        "weil die Risiken im Bankensektor gestiegen sind",
        "because risks in the banking sector increased",
        "by notifying the authority within four hours through the portal",
    ],
)
def test_an_explanation_carrying_a_verb_is_kept(answer) -> None:
    """The questions the one verb rule made unwritable.

    Measured before this existed: five of six realistic why, how and
    procedure answers were refused as malformed, and only a bare value like
    `EUR 15,000` survived. Every question in the set was a lookup because
    nothing else could be stored.
    """
    german = answer.startswith("weil")
    assert (
        checked(
            question_text=(
                "Warum wurde die Anforderung angehoben?"
                if german
                else "Why was the requirement raised?"
            ),
            target_answer=answer,
            language="de" if german else "en",
            form=AnswerForm.EXPLANATION,
        )
        is None
    )


def test_an_explanation_that_explains_nothing_is_refused() -> None:
    """A why answered with a noun phrase has not been answered."""
    failed = checked(
        question_text="Why was the requirement raised?",
        target_answer="the capital requirement of the banking sector",
        form=AnswerForm.EXPLANATION,
    )

    assert code(failed) == QuestionRejection.WRONG_FORM


def test_a_list_is_held_to_neither_verb_rule() -> None:
    """Items may be actions or things, and both are a list."""
    assert (
        checked(
            question_text="Which ways can a request be raised?",
            target_answer="by phone, through the web form and by email",
            form=AnswerForm.LIST,
        )
        is None
    )


def test_an_answer_over_the_ceiling_of_its_form_is_refused() -> None:
    """A value answered with a paragraph is the too-broad question again."""
    failed = checked(
        target_answer="4 kg, " * 30,
        form=AnswerForm.VALUE,
    )

    assert code(failed) == QuestionRejection.ANSWER_TOO_LONG


def test_the_same_answer_passes_for_one_form_and_fails_for_another() -> None:
    """Which is the whole point of the column: one rule fitted neither."""
    prose = "it is raised through the web form and confirmed by email"

    assert code(checked(target_answer=prose, form=AnswerForm.VALUE)) == (
        QuestionRejection.WRONG_FORM
    )
    assert checked(target_answer=prose, form=AnswerForm.EXPLANATION) is None


# ── Agreeing about an answer that is longer than a value ──────────────────


def test_a_paraphrased_explanation_is_recovered() -> None:
    """Demanding every word of prose survives a paraphrase refuses answers.

    The verifier plainly found this one; requiring a subset of lemmas, which
    is right for a value, threw it away.
    """
    assert agrees(
        "raised through the web form and confirmed by email",
        "it is raised through the web form, and confirmed by email before work begins",
        "en",
        AnswerForm.EXPLANATION,
        0.6,
    )


def test_an_explanation_about_something_else_is_not_recovered() -> None:
    """The floor still has to reject an answer that shares a few words."""
    assert not agrees(
        "the office is open on working days",
        "because the risk of a cyber incident rose sharply during the period",
        "en",
        AnswerForm.EXPLANATION,
        0.6,
    )


def test_a_number_still_has_to_match_exactly_whatever_the_form() -> None:
    """`4 hours` for `48 hours` is the failure the gate exists for."""
    assert not agrees(
        "answered within 4 hours on working days",
        "answered within 48 hours on working days",
        "en",
        AnswerForm.EXPLANATION,
        0.6,
    )


def test_a_value_is_still_compared_whole() -> None:
    """Every word of a value is the answer, so a share of it is not."""
    assert not agrees(
        "the supervisory authority", "the federal supervisory authority", "en"
    )


# ── The kind of question it was asked to be ───────────────────────────────


def test_no_gate_asks_the_verifier_what_kind_of_question_it_is_reading() -> None:
    """It was asked, and answered `true` for a question of the wrong kind.

    Measured over 71 questions the judgement fired zero times while the kind
    was plainly wrong on six of the 27 accepted - the same failure the
    phrasing judgement had while it was a yes or no. `wrong_form` carries the
    structural half, and `question_type` records what was asked for.
    """
    recording = Recording(recovers="because the plant was rebuilt in March")

    build(recording).check(
        candidate(
            question_text="Why was the device rebuilt at the Hamburg plant?",
            target_answer="because the plant was rebuilt in March",
            question_type="reason",
        )
    )

    assert recording.asked == ""


def test_the_type_and_the_form_are_stored_on_the_question() -> None:
    """A set cannot be filtered to its reasons unless each row says so."""
    result = build(Recording(recovers="4 kg")).check(candidate())

    assert result.question_type == "factoid"
    assert result.answer_form == "value"
    assert result.planned_difficulty == "easy"


def test_an_unanswerable_question_carries_no_answer_form() -> None:
    """It has no answer, so there is no shape for one to take."""
    result = build(Recording(recovers=None)).check(
        candidate(target_answer=None, answerable=False)
    )

    assert result.question_type == "factoid"
    assert result.answer_form is None


def test_the_verifier_is_told_to_answer_in_the_language_of_the_passages() -> None:
    """A translated answer agrees with nothing.

    Measured on one run: an 8B verifier answered a German question with `to
    extract confidential information from the training data` where the target
    was `Extraktion vertraulicher Informationen`. The two say the same thing
    and share not one lemma, so a correct question was rejected as
    not_recoverable.
    """
    from question_generation.verifier import _VERIFY

    assert "language" in _VERIFY


# ── Two questions welded with a conjunction ───────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        (
            "Welche Vorschriften gelten für das Kreditgeschäft und wie hoch ist "
            "die Gebühr für das inländische Investmentwesen?"
        ),
        (
            "Warum hat sich die Lage der Pensionskassen entspannt und wie wird "
            "die Kreditentwicklung eingeschätzt?"
        ),
        "What fee applies to a licence and who approves the shift plan?",
    ],
)
def test_a_question_asking_two_things_is_refused(question) -> None:
    """Real rows, all of them written into a wide sample.

    Told to use both passages of a pair with no single question between them,
    the writer welds two together. A chatbot answering one of them is neither
    right nor wrong, and nobody types this. 17 of the first 19 multi-passage
    questions had this shape.
    """
    failed = checked(
        question_text=question,
        target_answer=(
            "etwas und etwas anderes"
            if question.startswith(("Welche", "Warum"))
            else "a fee and a manager"
        ),
        language="de" if question.startswith(("Welche", "Warum")) else "en",
        form=AnswerForm.LIST,
    )

    assert code(failed) == QuestionRejection.COMPOUND


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Welche Arten von Kryptowerten gelten als reguliert?", "de"),
        (
            (
                "Wie viele Meldungen gab es insgesamt in den Jahren 2025 und "
                "2024 zusammen?"
            ),
            "de",
        ),
        ("How do the reply times for standard and urgent requests differ?", "en"),
        ("How is a support request raised and confirmed?", "en"),
    ],
)
def test_one_question_about_two_things_is_kept(question, language) -> None:
    """A conjunction is not two questions.

    A finite-verb count does not separate these: `Welche Arten von Kryptowerten
    gelten als reguliert?` carries two verbs and is one question, and a
    comparison names both sides by construction.
    """
    assert (
        checked(
            question_text=question,
            target_answer="two things, and another",
            language=language,
            form=AnswerForm.LIST,
        )
        is None
    )


def test_the_target_answer_may_restate_the_question_without_disagreeing() -> None:
    """A real rejection, and a false one.

    The writer answered `Die Regionen, in die chinesische Produkte exportiert
    werden, umfassen Südostasien und Afrika`; the verifier answered
    `Südostasien, Afrika und Europa`. They agree about everything the answer
    carries and share two lemmas of seven, so counting the question's own
    words threw a good question away.
    """
    question = "In welche Regionen werden chinesische Produkte exportiert?"
    target = (
        "Die Regionen, in die chinesische Produkte exportiert werden, "
        "umfassen Südostasien und Afrika."
    )

    assert not agrees("Südostasien, Afrika und Europa", target, "de", AnswerForm.LIST)
    assert agrees(
        "Südostasien, Afrika und Europa",
        target,
        "de",
        AnswerForm.LIST,
        OVERLAP,
        question,
    )


# ── The measurement behind the phrasing opinion ───────────────────────────


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("What specific components are included?", "en"),
        ("For which models do the requirements apply?", "en"),
        ("Which criteria are used?", "en"),
        ("Welche spezifischen Komponenten sind enthalten?", "de"),
    ],
)
def test_a_question_naming_nothing_is_thin_enough_to_reject(question, language) -> None:
    """The two the gate exists for, and two of the same shape."""
    assert not anchored(question, language)


@pytest.mark.parametrize(
    ("question", "language"),
    [
        (
            (
                "Wann wurde die Erlaubnis des Versicherers nach Eröffnung "
                "des endgültigen Insolvenzverfahrens widerrufen?"
            ),
            "de",
        ),
        ("Für welche Ziele setzt sich die Bafin ein?", "de"),
        ("Wozu nutzt die BaFin das DORA-Informationsregister?", "de"),
        ("What fee applies to a banking licence application?", "en"),
        ("Wie hoch war der Anteil der überschuldeten Verbraucher?", "de"),
    ],
)
def test_a_question_naming_something_is_never_called_unanchored(
    question, language
) -> None:
    """Real rows the verifier called unanchored, every one of them anchored.

    A name, a number, or three things. The first names five, the second and
    third name a party, and the gate may not overrule that.
    """
    assert anchored(question, language)


def test_the_verifier_cannot_reject_a_question_that_names_something() -> None:
    """An opinion needs a measurement behind it, not only a second model."""
    recording = Recording(recovers="4 kg", stands_alone=False)

    result = build(recording).check(
        candidate(question_text="What fee applies to a banking licence application?")
    )

    assert result.accepted, "the verifier overruled the question's own subjects"


def test_the_verifier_still_rejects_a_question_that_names_nothing() -> None:
    """Narrowing the gate must not turn it off."""
    recording = Recording(recovers="4 kg", stands_alone=False)

    result = build(recording).check(
        candidate(question_text="What specific components are included?")
    )

    assert result.rejected_reason == QuestionRejection.UNANCHORED


# ── The forms a type will take ─────────────────────────────────────────────


def test_a_type_takes_the_form_its_answer_actually_has() -> None:
    """A factoid answered with three tools is not a malformed factoid.

    The material decides what shape an answer has, not the question. Held to
    the one form its type asked for, `Welche Werkzeuge werden empfohlen?` ->
    `statische Analysatoren, Linter und Formatierer` was refused for the form
    it arrived in, and then held to a value's stricter comparison on the way
    out.
    """
    form, failed = fitting(
        "unterschiedliche Fähigkeiten, Kompetenzen, Rollen und Verantwortung",
        (AnswerForm.VALUE, AnswerForm.LIST),
        BOUNDS,
        "de",
    )

    assert failed is None
    assert form == AnswerForm.LIST


def test_a_condition_answered_with_a_value_is_kept() -> None:
    """`immer` is the condition, and it is one word."""
    form, failed = fitting("immer", (AnswerForm.LIST, AnswerForm.VALUE), BOUNDS, "de")

    assert failed is None
    assert form == AnswerForm.VALUE


def test_a_form_the_type_does_not_allow_is_still_refused() -> None:
    """The point is a wider set of forms, not no check at all."""
    _, failed = fitting(
        "die Testbarkeit des Systems", (AnswerForm.EXPLANATION,), BOUNDS, "de"
    )

    assert code(failed) == QuestionRejection.WRONG_FORM


def test_two_items_are_not_a_list() -> None:
    """`Berlin, 2025` is a value with a comma in it, not a set of two."""
    form, failed = fitting(
        "Berlin, 2025", (AnswerForm.VALUE, AnswerForm.LIST), BOUNDS, "en"
    )

    assert failed is None
    assert form == AnswerForm.VALUE


def test_the_complaint_reported_is_the_asked_for_forms() -> None:
    """That is the form the writer was told to produce, so it is the one to show."""
    form, failed = fitting("x" * 700, (AnswerForm.VALUE, AnswerForm.LIST), BOUNDS, "en")

    assert form == AnswerForm.VALUE
    assert code(failed) == QuestionRejection.ANSWER_TOO_LONG


# ── The gates an unanswerable question faces ───────────────────────────────


def test_a_question_about_the_material_is_on_topic() -> None:
    """A perturbation keeps the subject and moves one detail out of reach."""
    assert on_topic(
        "How long is allowed for a standard request on a public holiday?",
        "en",
        frozenset({"standard", "request", "holiday", "hour"}),
        0.3,
    )


def test_a_question_about_nothing_the_material_mentions_is_off_topic() -> None:
    """Any chatbot declines it, so declining it proves nothing."""
    assert not on_topic(
        "What is the top speed of a swallow?",
        "en",
        frozenset({"standard", "request", "holiday", "hour"}),
        0.3,
    )


def test_a_passage_with_no_lemmas_refuses_nothing() -> None:
    """A measurement with nothing to measure against is not evidence."""
    assert on_topic("What is the top speed of a swallow?", "en", frozenset(), 0.3)


def test_a_floor_of_zero_turns_the_gate_off() -> None:
    """One setting, and 0 is how a deployment declines to hold the opinion."""
    assert on_topic("What is the top speed of a swallow?", "en", frozenset({"a"}), 0.0)


def test_an_unanswerable_question_another_passage_answers_is_refused() -> None:
    """The one claim here that is about the corpus and not about two passages.

    A question wrongly carrying "nothing here answers this" marks a correct
    chatbot wrong, which is the failure every gate exists to stop.
    """
    recording = Recording(recovers=None)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        bounds=BOUNDS,
        elsewhere=lambda lemmas, language, skip, limit, embedding=None: [
            "It is 48 hours."
        ],
        elsewhere_passages=4,
    )

    # The corpus probe re-reads with the same stub, which recovers nothing on
    # the first pass; scripting it to recover is what the second pass reads.
    recording.recovers = None
    result = checker.check(
        candidate(
            question_text="How long is allowed for a request on a holiday?",
            target_answer=None,
            answerable=False,
        )
    )

    # Nothing was found anywhere, so it stands.
    assert result.status == "accepted"


# ── The entailment pass ────────────────────────────────────────────────────


def test_recall_missing_an_answer_the_passages_support_is_rescued() -> None:
    """Recalling an answer cold is a harder task than checking one.

    Over one corpus the verifier failed to recall an answer 76 times against
    19 real disagreements about what the answer was, so four in five of the
    biggest rejection class were the task and not the question.
    """
    recording = Recording(recovers=None, backs=True)

    result = build(recording).check(candidate(target_answer="4 kg"))

    assert result.status == "accepted"
    assert recording.supported == 1


def test_the_entailment_pass_can_only_accept() -> None:
    """A question reaching it was already being refused."""
    recording = Recording(recovers=None, backs=False)

    result = build(recording).check(candidate(target_answer="4 kg"))

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
    assert recording.supported == 1


def test_an_answer_to_a_different_question_is_not_rescued() -> None:
    """The LMT case, which the rescue pass used to wave through.

    `Was ist ein Liquiditätsmanagementtool?` answered `eine einjährige
    Rückgabefrist` clears the numbers guard, because that phrase is lifted
    straight out of the passage, and any judge asked "do these passages say
    what this answer says" agrees - they do say it. What they never mention
    is a Liquiditätsmanagementtool, so the answer is supported and answers
    something else.
    """
    recording = Recording(recovers=None, backs=True)

    result = build(recording).check(
        candidate(
            question_text="Was ist ein Liquiditätsmanagementtool?",
            target_answer="eine einjährige Rückgabefrist",
            facts=group(
                source(
                    1,
                    statement="Für den Fonds gilt eine einjährige Rückgabefrist.",
                    language="de",
                    passage_text=(
                        "Für den Fonds gilt eine einjährige Rückgabefrist. Die "
                        "Verwaltungsgesellschaft veröffentlicht die Frist im "
                        "Verkaufsprospekt."
                    ),
                )
            ),
        )
    )

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
    assert recording.supported == 0, "the judgement is never reached"


class Reader:
    """An extractive reader answering with one span, or with none.

    `only_in` narrows it to passages carrying that text, which is how a
    test tells the cited passages from the ones the corpus probe found.
    """

    def __init__(self, text: str | None, only_in: str | None = None) -> None:
        self.text = text
        self.only_in = only_in
        self.asked = 0

    def answer(self, question: str, passages, confidence: float):
        """The scripted span, and a count of the asks."""
        from nlp.qa import Span

        self.asked += 1
        if self.text is None:
            return None
        if self.only_in is not None and not any(
            self.only_in in passage for passage in passages
        ):
            return None
        return Span(self.text, 0.95, 0)


def _with_reader(recording: Recording, reader: Reader) -> QuestionChecker:
    """A checker whose recall half asks the encoder first."""
    return QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        phrasing=recording,
        extractive=reader,
    )


def test_a_confident_span_answers_recall_without_a_call() -> None:
    """The easy majority, answered locally in milliseconds."""
    recording = Recording(recovers="4 kg")
    reader = Reader("4 kg")

    result = _with_reader(recording, reader).check(candidate(target_answer="4 kg"))

    assert result.accepted
    assert reader.asked == 1
    assert recording.verified == 0, "the served model was not needed"


def test_no_span_falls_through_to_something_that_can_read() -> None:
    """Every span it gives is contiguous and inside one passage.

    The verifier is allowed to put an answer together from two sentences,
    so a reader that found nothing must not be read as a refusal.
    """
    recording = Recording(recovers="4 kg")
    reader = Reader(None)

    result = _with_reader(recording, reader).check(candidate(target_answer="4 kg"))

    assert result.accepted
    assert reader.asked == 1
    assert recording.verified == 1, "the served model answered instead"


def test_a_follow_up_never_reaches_the_extractive_reader() -> None:
    """`And for an urgent one?` is answered in a conversation.

    An extractive model has nowhere to put one, so it would answer the
    question as if it had been asked cold.
    """
    recording = Recording(recovers="4 hours")
    reader = Reader("4 kg")

    _with_reader(recording, reader).check(
        candidate(
            question_text="How long is allowed for answering an urgent request?",
            target_answer="4 hours",
            thread=(("How long for a standard support request?", "48 hours"),),
        )
    )

    assert reader.asked == 0
    assert recording.verified == 1


def test_an_encoder_answers_the_pass_where_one_is_configured() -> None:
    """The served model is not asked when an NLI model is to hand."""

    class Encoder:
        """One verdict, and a count of the pairs it was handed."""

        def __init__(self, entailment: float) -> None:
            self.entailment = entailment
            self.pairs: list[tuple[str, str]] = []

        def judge_all(self, pairs):
            from nlp.entailment import Verdict

            self.pairs.extend(pairs)
            return [Verdict(self.entailment, 1 - self.entailment, 0.0) for _ in pairs]

    recording = Recording(recovers=None, backs=False)
    encoder = Encoder(entailment=0.9)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        phrasing=recording,
        entailment=encoder,
        entailment_threshold=0.5,
    )

    result = checker.check(candidate(target_answer="4 kg"))

    assert result.accepted, "the encoder said the passages entail it"
    assert recording.supported == 0, "no served model was asked"
    assert encoder.pairs, "the encoder was handed a premise and a hypothesis"


def test_the_encoder_is_asked_one_passage_at_a_time() -> None:
    """One premise, one hypothesis, which is what an NLI model was trained on.

    A premise of several passages joined is both a worse question than the
    one being asked and longer than some heads accept.
    """

    class Encoder:
        """Refuses everything, and remembers what it was shown."""

        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def judge_all(self, pairs):
            from nlp.entailment import Verdict

            self.pairs.extend(pairs)
            return [Verdict(0.0, 1.0, 0.0) for _ in pairs]

    recording = Recording(recovers=None, backs=False)
    encoder = Encoder()
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        phrasing=recording,
        entailment=encoder,
    )

    checker.check(candidate(target_answer="4 kg"))

    assert len(encoder.pairs) == 1, "one pair per passage, and there is one"
    assert "\n\n" not in encoder.pairs[0][0], "the passages were joined"


def test_a_number_the_passages_never_gave_is_not_rescued() -> None:
    """The one error that must not get through, whatever the model says.

    `4 hours` confirmed against a passage that says 48 puts a question in
    the benchmark whose answer the material does not give, which marks a
    correct chatbot wrong.
    """
    recording = Recording(recovers=None, backs=True)

    result = build(recording).check(
        candidate(
            question_text="How long is allowed for a standard request?",
            target_answer="48 hours",
            facts=group(
                source(passage_text="A standard request is answered within 4 hours.")
            ),
        )
    )

    assert result.rejected_reason == QuestionRejection.NOT_RECOVERABLE
    # Refused before the model was asked, because the guard is free.
    assert recording.supported == 0


def test_a_target_asserting_nothing_numeric_reaches_the_pass() -> None:
    """The guard is about numbers and names; prose has none to check."""
    recording = Recording(recovers=None, backs=True)

    result = build(recording).check(
        candidate(
            question_text="Why must a request be confirmed in writing?",
            target_answer="so that the agreed time can be evidenced later",
            question_type=QuestionType.REASON,
            facts=group(
                source(
                    passage_text=(
                        "Requests are confirmed in writing so the agreed time "
                        "can be evidenced later."
                    )
                )
            ),
        )
    )

    assert result.status == "accepted"
    assert recording.supported == 1


# ── Pointing at what the asker cannot see ─────────────────────────────────


def test_a_question_pointing_outside_itself_is_refused() -> None:
    """The failure that shipped: correct arithmetic, unanswerable question.

    `Wie groß ist der Unterschied ... zwischen diesen beiden Lehrplänen?` was
    accepted over two facts that do differ by the answer it gives. Nobody can
    answer it, because nothing says which two.
    """
    recording = Recording(recovers="2.5 hours", self_contained=False)

    result = build(recording).check(
        candidate(question_text="How long is the gap between these two editions?")
    )

    assert result.rejected_reason == QuestionRejection.UNANCHORED


def test_a_question_that_sets_its_own_case_up_is_kept() -> None:
    """The measurement over-fires and may only veto, so the verifier decides.

    Half the questions carrying a pointer name what it points at first. They
    are the `application` type working as designed, and a rule reading the
    pointer alone would throw every one of them out.
    """
    recording = Recording(recovers="4 kg", self_contained=True)

    result = build(recording).check(
        candidate(
            question_text=(
                "If a system meets its target by editing the stored score, "
                "how is this behaviour classified?"
            )
        )
    )

    assert result.status == QuestionStatus.ACCEPTED


def test_a_follow_up_may_point_at_the_conversation() -> None:
    """Leaning on the thread is what a follow-up is for."""
    recording = Recording(recovers="4 kg", self_contained=False)

    result = build(recording).check(
        candidate(
            question_text="And how long does that one take?",
            thread=(("What does the device weigh?", "4 kg"),),
        )
    )

    assert result.rejected_reason != QuestionRejection.UNANCHORED


def test_a_reader_finding_nothing_elsewhere_saves_the_call() -> None:
    """The gate fired 8 times in 3,131, so nearly every call confirms nothing.

    There is no target answer here, which is why an extractive reader
    belongs in this gate and not in the recall half: the only question is
    whether some passage answers this at all.
    """
    # The cited passages do not answer it, so the gate before this one
    # passes and `_corpus` is reached.
    recording = Recording(recovers=None)
    reader = Reader(None)
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        phrasing=recording,
        elsewhere=lambda *a, **k: ["Something else entirely."],
        elsewhere_passages=2,
        extractive=reader,
    )

    result = checker.check(candidate(target_answer=None, answerable=False))

    assert result.accepted
    # Once, here. The recall half of an unanswerable question does not ask
    # it: a span there is a rejection, which is not an extractor's to make.
    assert reader.asked == 1
    assert recording.verified == 1, "only the recall call, not the corpus one"


def test_a_span_found_elsewhere_still_goes_to_the_verifier() -> None:
    """Rejecting here marks a correct chatbot wrong.

    That verdict is not one an extractor should make alone, so a span is
    a reason to ask rather than an answer.
    """
    recording = Recording(recovers=None)
    # Found only in the passage the corpus probe returns, never in the ones
    # the question cites - which is what this gate is about. The verifier
    # reads the same way, so the cited passages answer nothing and the
    # uncited one answers.
    reader = Reader("4 kg", only_in="uncited note")

    def read(question, passages, thread=()):
        recording.verified += 1
        found = any("uncited note" in passage for passage in passages)
        return Reading(recovered="4 kg" if found else None)

    recording.read = read  # type: ignore[method-assign]
    checker = QuestionChecker(
        embedder=recording,
        verifier=recording,
        nearest=lambda embedding: None,
        threshold=0.93,
        phrasing=recording,
        elsewhere=lambda *a, **k: ["An uncited note records 4 kg."],
        elsewhere_passages=2,
        extractive=reader,
    )

    result = checker.check(candidate(target_answer=None, answerable=False))

    assert result.rejected_reason == QuestionRejection.ANSWERABLE_ELSEWHERE
    assert recording.verified == 2, "the recall call, and the verifier confirming"


def test_a_span_in_its_own_passages_does_not_reject_an_unanswerable_question() -> None:
    """The same verdict as the gate above, so the same rule applies.

    `answerable_after_all` says the cited passages answer a question
    written to have none, which marks a correct chatbot wrong. The reader
    was deciding that alone, where one passage further on it may only ask.
    """
    recording = Recording(recovers=None)
    reader = Reader("4 kg")

    result = _with_reader(recording, reader).check(
        candidate(target_answer=None, answerable=False)
    )

    assert result.accepted
    assert reader.asked == 0, "the recall half of an unanswerable one is the verifier's"
    assert recording.verified == 1


def test_the_verifier_still_rejects_an_unanswerable_question_it_can_answer() -> None:
    """Removing the reader from this path must not remove the gate."""
    recording = Recording(recovers="4 kg")

    result = _with_reader(recording, Reader("4 kg")).check(
        candidate(target_answer=None, answerable=False)
    )

    assert result.rejected_reason == QuestionRejection.ANSWERABLE_AFTER_ALL


# ── The long answer ────────────────────────────────────────────────────────


#: A passage holding every figure the explanations below assert, so a test
#: about length or restatement is not stopped by `asserted`.
_PASSAGES = (
    (
        "A standard support request is answered within 48 hours. An urgent "
        "request is answered within 4 hours, and only on working days."
    ),
)


def _reads(text: str) -> bool:
    """Whether one explanation of the 48-hour answer is usable."""
    return explains(text, "48 hours", _PASSAGES, "en")


def test_no_explanation_at_all_is_not_a_failure() -> None:
    """The column is nullable, and a row written before it has none."""
    assert _reads("")
    assert explains("", "48 hours", _PASSAGES, "en")


def test_an_explanation_too_short_to_be_a_reading_is_refused() -> None:
    """This is the failure the column was added for: the key said twice."""
    assert not _reads("It is 48 hours.")


def test_an_explanation_past_its_ceiling_is_refused() -> None:
    """An explanation that long has become the passage."""
    assert not explains("word " * 400, "48 hours", _PASSAGES, "en", (150, 900))


def test_an_explanation_asserting_a_figure_the_passages_lack_is_refused() -> None:
    """The one failure here that can put a wrong claim before a reader.

    Everything else this gate catches is a bad read. A number the material
    never gave is the defect `asserted` exists for, and 72 is not in the
    passages.
    """
    assert not _reads(
        "A standard support request has to be answered within 72 hours. That "
        "is the ordinary service level and the slower of the two the material "
        "sets, because an urgent request is answered far sooner than a "
        "standard one ever is."
    )


def test_an_explanation_that_only_restates_the_key_is_refused() -> None:
    """Padding the key to length is not explaining it."""
    assert not explains(
        "48 hours. 48 hours. 48 hours. 48 hours. 48 hours. 48 hours. 48 "
        "hours. 48 hours. 48 hours. 48 hours. 48 hours. 48 hours. 48 hours.",
        "48 hours",
        _PASSAGES,
        "en",
    )


def test_an_explanation_that_reads_and_rests_on_the_passages_passes() -> None:
    """What the column is for."""
    assert _reads(
        "A standard support request has to be answered within 48 hours. That "
        "is the ordinary service level, and it is the slower of the two the "
        "material sets: a request marked urgent is answered within 4 hours "
        "instead. The hours are the time allowed for the answer rather than "
        "for resolving what was asked about."
    )


# ── The kind a question turned out to be ───────────────────────────────────


def test_an_entity_question_asking_who_names_a_party() -> None:
    """The interrogative settles it without reading the answer."""
    assert asks_for_an_agent("Who signs off a change to the shift plan?", "", "en")


def test_an_entity_question_whose_answer_names_an_organisation_passes() -> None:
    """The other way to pass: the party is in the answer, not the question."""
    assert asks_for_an_agent(
        "Which body accredits a training provider?", "the ISTQB", "en"
    )


def test_an_entity_question_asking_after_no_party_is_refused() -> None:
    """85% of one run's accepted entity questions read like this one."""
    assert not asks_for_an_agent(
        "Womit können Test-Chartas erstellt werden?",
        "mit Flipcharts und Tabellenkalkulationen",
        "de",
    )


def test_an_enumeration_answering_with_one_thing_is_refused() -> None:
    """Its own directive says so, and nothing enforced it."""
    assert not enumerates("an ISTQB examination", "en")


def test_an_enumeration_answering_with_a_set_passes() -> None:
    """Counted on noun phrases, so the punctuation does not decide."""
    assert enumerates("by phone, through the web form and by email", "en")


# ── A follow-up, read on its facts rather than its words ───────────────────


def test_a_follow_up_citing_only_its_root_asks_nothing_new() -> None:
    """70.6% of one run's accepted follow-ups were this."""
    assert not moves_on(cited=(1, 2), root=(1, 2, 3))


def test_a_follow_up_reaching_one_new_fact_moves_on() -> None:
    """One is enough: the thread went somewhere."""
    assert moves_on(cited=(2, 4), root=(1, 2))


def test_a_follow_up_sharing_a_passage_stays_on_thread() -> None:
    """What makes it the next turn rather than another question."""
    assert same_material(cited=(7, 8), parent=(8, 9))


def test_a_follow_up_sharing_no_passage_has_changed_the_subject() -> None:
    """`Warum wurde TTA-2.6.1 entfernt?` after a question about test design."""
    assert not same_material(cited=(7,), parent=(8, 9))


def test_a_turn_resting_on_nothing_is_left_to_the_orphan_trigger() -> None:
    """A measurement with nothing to measure is not evidence."""
    assert same_material(cited=(), parent=(8,))
    assert same_material(cited=(8,), parent=())


# ── The answer that only says the question again ───────────────────────────


def test_an_answer_adding_no_content_word_restates_the_question() -> None:
    """3.3% of one run's accepted answers read like this."""
    assert restates(
        "Wofür sind statische Analysen bei statischen Wartbarkeitstests geeignet?",
        "für statische Wartbarkeitstests",
        "de",
    )


def test_an_answer_that_adds_something_does_not_restate() -> None:
    """The gate is full containment, not a share."""
    assert not restates(
        "Why does a request have to be confirmed in writing?",
        "so that the agreed response time can be evidenced later",
        "en",
    )


def test_an_answer_asserting_a_number_is_exempt() -> None:
    """`content` drops numerals, so the answer would score as contained.

    `zwei` IS the answer, and a value carrying a figure is the most
    scoreable kind there is. The exemption took the catch from 167 answers
    to 121 and every one it dropped was a number.
    """
    assert not restates(
        "In wie viele Kategorien werden Schlüsselwörter eingeteilt?",
        "zwei Kategorien",
        "de",
    )


# ── The key that names less than the passages gave ─────────────────────────


def test_a_key_naming_one_of_many_is_incomplete() -> None:
    """The `mit Entwicklern` case, read off the recovery already paid for."""
    assert incomplete(
        recovered=(
            "developers, architects, operations engineers, product owners, "
            "local support, technical experts and the service desk"
        ),
        target="developers",
        language="en",
        form=AnswerForm.LIST,
    )


def test_a_key_naming_most_of_what_came_back_is_complete() -> None:
    """A verifier is wordier than a key, and that alone is not a fault."""
    assert not incomplete(
        recovered="by phone, through the web form and by email",
        target="by phone, through the web form and by email",
        language="en",
        form=AnswerForm.LIST,
    )


@pytest.mark.parametrize("form", [AnswerForm.VALUE, AnswerForm.EXPLANATION])
def test_only_a_list_claims_to_name_a_set(form: str) -> None:
    """A value is short because its answer is one thing."""
    assert not incomplete(
        recovered=(
            "developers, architects, operations engineers, product owners, "
            "local support, technical experts and the service desk"
        ),
        target="developers",
        language="en",
        form=form,
    )


# ── Two more types a rule can settle ───────────────────────────────────────


def test_a_comparison_naming_one_thing_is_refused() -> None:
    """`Wie unterscheiden sich die beschriebenen Umgebungen?` compares nothing."""
    assert not compares(
        "Wie unterscheiden sich die beschriebenen Umgebungen?",
        "eine Zielumgebung",
        "de",
    )


def test_a_comparison_naming_both_sides_passes() -> None:
    """Both in the question, both in the answer."""
    assert compares(
        "How do the reply times for standard and urgent requests differ?",
        "48 hours for a standard request and 4 hours for an urgent one",
        "en",
    )


def test_a_temporal_answer_needs_two_periods() -> None:
    """`How did it change from 2024 to 2025?` is answered with both years."""
    assert periods("from 72 hours in 2024 to 48 hours in 2025", "en")
    assert not periods("72 hours", "en")


def test_a_count_is_not_a_period() -> None:
    """Two units are not two periods; a quantity is a unit too."""
    assert not periods("40 people and 25 people", "en")


# ── The corpus's other furniture ───────────────────────────────────────────


def test_a_page_of_names_is_read_as_a_credits_page() -> None:
    """Acknowledgements: median PER density 0.794 over 20 of them."""
    assert names_parties(
        "Graham Bath, Judy McKay, Tauhida Parveen, Mike Smith, Erik van "
        "Veenendaal, Rex Black, Kari Kakkonen, Leo van der Aalst",
        "en",
    )


def test_subject_matter_is_not_a_credits_page() -> None:
    """Median PER density of everything else was 0.000."""
    assert not names_parties(
        "A standard support request is answered within 48 hours on working "
        "days, and an urgent request within four hours of being raised.",
        "en",
    )


def test_a_density_of_zero_turns_the_reading_off() -> None:
    """The default, so a caller that says nothing gets every passage."""
    assert not names_parties("Graham Bath, Judy McKay, Rex Black", "en", 0.0)
