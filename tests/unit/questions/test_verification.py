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
            "min_answer_chars": 0,
            "titles": (),
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
    ):
        """Initialises with what to answer, and nothing asked yet."""
        self.vector = vector or [1.0] + [0.0] * 1023
        self.recovers = recovers
        self.stands_alone = stands_alone
        self.names_its_source = names_its_source
        self.threaded: tuple = ()
        self.embedded = 0
        self.verified = 0

    def embed(self, text: str) -> list[float]:
        """Answers with the one vector, and counts the ask."""
        self.embedded += 1
        return self.vector

    def read(self, question: str, passages, thread=()) -> Reading:
        """Answers with the scripted reading, and counts the ask."""
        self.verified += 1
        self.threaded = tuple(thread)
        return Reading(
            recovered=self.recovers,
            stands_alone=self.stands_alone,
            names_its_source=self.names_its_source,
        )


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
    failed = checked(target_answer=answer, min_answer_chars=15)

    assert code(failed) == QuestionRejection.ANSWER_TOO_SHORT


@pytest.mark.parametrize(
    "answer", ["European Supervisory Authorities", "29 January 2027", "Article 4(1)(1)"]
)
def test_an_answer_over_the_floor_is_kept(answer) -> None:
    """A name, a date and a citation, all of them scoreable."""
    assert checked(target_answer=answer, min_answer_chars=15) is None


def test_a_floor_of_zero_refuses_nothing_for_length() -> None:
    """Which is what keeps the bare values a corpus of numbers needs."""
    assert checked(target_answer="70%", min_answer_chars=0) is None


def test_an_unanswerable_question_is_never_too_short() -> None:
    """It has no answer to measure, and is scored on behaviour."""
    assert checked(answerable=False, target_answer=None, min_answer_chars=15) is None


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

    result = build(recording).check(candidate())

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
    The model half is what the paraphrase is for.
    """
    recording = Recording(recovers="4 kg", names_its_source=True)

    result = build(recording).check(candidate())

    assert result.rejected_reason == QuestionRejection.LEAKS_SOURCE
    assert recording.verified == 1, "it must not cost a second call"


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
