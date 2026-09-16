"""The gates a written question has to pass.

Applied cheapest first, because each one that fires saves the cost of those
behind it:

  malformed       structural, free
  answer_too_short  free: nothing worth scoring against
  answer_too_long   free: not the form of answer that was asked for
  wrong_form        free: a value carrying a verb, an explanation with none
  leaks_source    free where the title is quoted, the verifier's otherwise
  duplicate       one index probe against the questions already accepted
  answerable      the same probe, when an unanswerable question has a twin
                  the corpus does answer
  compound        it asks two things, so half an answer is neither right
                  nor wrong
  unanchored      a question nobody could have asked without the passage
  recoverable     the answer is not in the evidence the question cites

The last three come out of one model call.

There is no gate here for "it is not the kind of question it was asked to be",
and that is a finding rather than an omission. The verifier was asked it, as
`matches_intent`, and answered `true` for a bare `Wie viele Anlassprüfungen
wurden 2025 durchgeführt?` written into a `reason` slot - the same failure the
phrasing judgement had when it was a yes or no. What survives is structural:
a kind declares an answer form, and `wrong_form` reads the answer against it,
so a reason answered with a value is still refused. `question_type` records
what was ASKED for, as `planned_difficulty` does. Recoverability is the one no
similarity measure makes: it asks whether the answer can be got back out of
the passages, which is what a question is for, and a paraphrase, a
decomposition and a resolved pronoun all survive it where a threshold would
not.

Every gate that reads an answer reads it against the FORM the question's type
asked for. One rule for all of them was what made the set what it was: a value
may carry no verb, so applied to every answer that rule refused every why,
how, procedure and consequence question ever written.

Two things are load-bearing. The verifier is a different model from the
writer, because a model marking its own work agrees with itself. And it sees
only the cited passages, never the corpus: what is being measured is the
dataset, not a retriever.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from database.qa_generator import AnswerForm, QuestionRejection, QuestionStatus
from llm.client import Client
from nlp.analysis import claim, content, interrogatives, normalised, vocabulary
from nlp.language import detect
from question_generation.embedding import Embedder, cosine
from question_generation.models import (
    LONG_ANSWER_CHARS,
    Candidate,
    CheckedQuestion,
    Neighbour,
)

#: Length bounds per answer form, as QUESTIONS_ANSWER_CHARS sets them. This is
#: the fallback for a re-check reading a row whose settings are not to hand,
#: as LONG_ANSWER_CHARS is.
BOUNDS: dict[str, tuple[int, int]] = {
    AnswerForm.VALUE: (1, 80),
    AnswerForm.LIST: (3, 300),
    AnswerForm.EXPLANATION: (20, 600),
}

#: How much of what the target answer is about has to occur in what the
#: verifier recovered, for a list or an explanation. A value is compared
#: whole, because every word of one is the answer.
OVERLAP = 0.6

log = logging.getLogger(__name__)

_VERIFY = """You check one test question for a document-search chatbot.

You are given some PASSAGES and a QUESTION. Do two separate things.

FIRST, answer the question using ONLY those passages. You have no other
knowledge. Nothing you know from anywhere else counts, and nothing you can
work out counts either: if the passages do not state it, it is not there.

- `in_passage` is true ONLY when the passages state the answer outright. If
  you are completing, inferring, rounding or assuming, it is false.
- `answer` is the answer, taken from the passages, when in_passage is true.
  Leave it empty otherwise. IN THE WORDS OF THE PASSAGES, and so in their
  language: an answer translated into English is compared against a German
  target and agrees with nothing.

Saying it is not in the passages is a correct answer and is the one we are
looking for whenever it is true. Guessing is the failure.

SECOND, two narrow readings of the question itself. They are about different
things and are easy to confuse, so read both.

- `names_its_source`: does the question say WHERE the answer is - naming or
  quoting a document, a report, a circular, a regulation by name, a section or
  a heading?

  TRUE:  "Laut den 'Risiken im Fokus 2026', wie viele Risiken werden genannt?"
  TRUE:  "Gemäß der MaRisk, wie viele Modelltypen sind betroffen?"
  TRUE:  "According to the annual report, what were the fees?"
  TRUE:  "Under 'Fees > Banking', what is the charge?"
  FALSE: "Wie viele Modelltypen unterliegen den Anforderungen?"
  FALSE: "What fee applies to a banking licence application?"
  FALSE: "How many cyber incidents were reported to BaFin in 2025?"
                          (BaFin is who they were reported TO, not a source)

  Naming a party, a duty, a period or a thing being regulated is NOT naming a
  source. Only saying which material holds the answer is.

- `subject`: COPY the thing the question is about, word for word out of the
  question. Not what it asks for - what it asks ABOUT.

  "What fee applies to a banking licence application?"  -> "banking licence
                                                            application"
  "How many incidents were reported in 2025?"           -> "incidents
                                                            reported in 2025"
  "What specific components are included?"              -> ""
  "For which models do the requirements apply?"         -> ""

  Leave it EMPTY only when the question names nothing at all - when every
  noun in it is a bare word like "components", "requirements" or "criteria"
  with nothing saying whose or which. If you can copy anything more specific
  than that out of the question, copy it.

  A question you could not answer still has a subject. Not finding the answer
  in the passages says nothing about this, and the two are constantly
  confused: answer this one by reading the QUESTION, not the passages.

All the judgements are independent. A question can be answerable by the
passages, name no source, and still name no subject either.
"""


class _Recovered(BaseModel):
    """What the verifier got back, and what it made of the question."""

    in_passage: bool = Field(
        description="True only when the passages state the answer outright. "
        "False is NOT IN PASSAGE, and is a correct answer."
    )
    answer: str = Field(
        default="",
        description="The answer, taken from the passages and in their "
        "language. Empty when in_passage is false.",
    )
    names_its_source: bool = Field(
        default=False,
        description="True if the question says WHERE the answer is: naming or "
        "quoting a document, report, circular, named regulation, section or "
        "heading. Naming a party, duty, period or regulated thing is not "
        "naming a source.",
    )
    subject: str = Field(
        default="",
        description="The thing the question is ABOUT, copied word for word "
        "out of the question. Empty only when it names nothing at all. Read "
        "the question, not the passages: a question you could not answer "
        "still has a subject.",
    )


#: The shortest document title worth matching a question against. Below it a
#: title is a word rather than a name - this corpus has `Contents` - and a
#: question may contain one without citing anything.
_TITLE_CHARS = 12


def _bare(text: str) -> str:
    """Folds a string for comparison, sentence punctuation and all."""
    return normalised(text).rstrip("?.!")


def _folded(token: str) -> str:
    """Drops the separators that only say which locale wrote a number.

    `0,3` and `0.3` are one value; so are `1.033` and `1,033`. Only applied
    when comparing one number against another, so no word is affected.
    """
    return token.replace(",", "").replace(".", "")


def structural(
    *,
    question_text: str,
    target_answer: str | None,
    answerable: bool,
    language: str | None,
    statements: Sequence[str] = (),
    form: str = AnswerForm.VALUE,
    bounds: Mapping[str, tuple[int, int]] | None = None,
    titles: Sequence[str] = (),
) -> tuple[str, str] | None:
    """The gates that need neither a model nor an index, or None if it passes.

    Returns the code to store and the reason to log. The several ways of
    being malformed share one code, because none of them measures anything a
    report would group on and the reason belongs where a person debugging
    reads it. The ones that do get a code of their own - an answer outside the
    bounds of its form, an answer of the wrong form, and a question quoting
    the title of its own source - are the ones a report should count.

    `form` is the shape the question's type asked its answer to take, and
    every check on the answer reads it.
    """
    low, high = (bounds or BOUNDS).get(form, BOUNDS[AnswerForm.VALUE])
    asked = question_text.strip()
    if not asked:
        return QuestionRejection.MALFORMED, "the model returned no question"
    if not asked.endswith("?"):
        return QuestionRejection.MALFORMED, "it does not end in a question mark"
    if asked.count("?") > 1:
        return (
            QuestionRejection.MALFORMED,
            (
                f"it asks {asked.count('?')} things; a chatbot answering one of "
                f"them is neither right nor wrong"
            ),
        )
    if answerable and not (target_answer or "").strip():
        return (
            QuestionRejection.MALFORMED,
            (
                "it is answerable and carries no target answer, so nothing could "
                "be scored against it"
            ),
        )
    if not answerable and (target_answer or "").strip():
        return (
            QuestionRejection.MALFORMED,
            (
                "it is unanswerable and carries a target answer; an unanswerable "
                "question is scored on behaviour, not on content"
            ),
        )
    # Without the trailing punctuation, which is the only thing a fact handed
    # back as a question differs from the fact by - and handing it back is
    # the laziest thing the writer can do with the prompt.
    if any(_bare(asked) == _bare(one) for one in statements):
        return (
            QuestionRejection.MALFORMED,
            (
                "it is one of its own facts handed back rather than a question "
                "written from one"
            ),
        )

    # Two question words is two questions welded with a conjunction:
    # `Welche Vorschriften gelten für X und wie hoch ist die Gebühr für Y?`.
    # A chatbot answering half of it is neither right nor wrong, and nobody
    # types it. This is what a writer told to use both passages of a wide
    # sample does when the two have no single question between them - 17 of
    # the first 19 multi-passage questions were this shape, and every one was
    # rejected downstream for an answer the verifier could only half recover.
    asked_about = interrogatives(asked, language)
    if len(asked_about) > 1:
        return (
            QuestionRejection.COMPOUND,
            (
                f"it asks {len(asked_about)} things ({', '.join(asked_about)}); "
                f"a chatbot answering one of them is neither right nor wrong"
            ),
        )

    # Bounds per form, so a question is not scored against something too thin
    # to be worth scoring - `7`, `8%`, `Nein` - nor against an essay where a
    # value was asked for. QUESTIONS_ANSWER_CHARS sets both ends of each form.
    answer = (target_answer or "").strip()
    if answer and len(answer) < low:
        return (
            QuestionRejection.ANSWER_TOO_SHORT,
            (
                f"the target answer {answer!r} is {len(answer)} characters and a "
                f"{form} answer is held to at least {low}"
            ),
        )
    if answer and len(answer) > high:
        return (
            QuestionRejection.ANSWER_TOO_LONG,
            (
                f"the target answer {answer!r} is {len(answer)} characters and a "
                f"{form} answer is held to at most {high}"
            ),
        )

    # The form the type asked for, read off the answer's parse.
    #
    # A value carrying a verb is a fragment of the sentence rather than the
    # thing asked for: `Verwarnungen aussprechen`, `nachvollziehbar zu
    # begründen`. Any verb and not only a finite one, because those two are
    # infinitives and make no claim - a phrase can describe an action without
    # asserting it. Only for an answer of more than one word: a single word is
    # a thing by construction, and `de_core_news_md` tags `dreihundert` as a
    # verb.
    #
    # An explanation carrying none is the opposite failure: a why or a how
    # answered with a noun phrase has not been answered.
    if answer and form == AnswerForm.VALUE:
        if len(answer.split()) > 1 and claim(answer, language).verbs:
            return (
                QuestionRejection.WRONG_FORM,
                (
                    f"the target answer {answer!r} describes an action rather "
                    f"than naming a thing, and this question asked for a value"
                ),
            )
    elif (
        answer and form == AnswerForm.EXPLANATION and not claim(answer, language).verbs
    ):
        return (
            QuestionRejection.WRONG_FORM,
            (
                f"the target answer {answer!r} names a thing rather than "
                f"explaining anything, and this question asked for an explanation"
            ),
        )

    # The question quotes the title of the document it came from, which is
    # the free half of the source-leak check. Verbatim and whole, and only
    # for a title long enough to be a title: `Contents` and `March 2018` are
    # rows in this corpus's documents table, and a question is allowed to
    # contain either by accident. `Risiken im Fokus 2026` it is not.
    #
    # The model catches the rest - `Laut dem Jahresbericht 2025` names a
    # source without quoting `Druckversion - Jahresbericht 2025` exactly -
    # but this half costs nothing and runs before any call.
    folded = normalised(asked)
    quoted = [
        title
        for title in titles
        if len(title.strip()) >= _TITLE_CHARS and normalised(title) in folded
    ]
    if quoted:
        return (
            QuestionRejection.LEAKS_SOURCE,
            (
                f"it names the document it came from ({quoted[0]!r}); nobody asks "
                f"a question while saying which file holds the answer"
            ),
        )

    # There is no gate here for "the question is its own fact rearranged",
    # and that is a finding rather than an omission. One was written, in two
    # formulations, and measured against 61 real rows: both refused questions
    # like `Wie hoch war die Arbeitslosenquote im August 2025?` -> `6,4
    # Prozent`, which is as good as a benchmark question gets.
    #
    # The reason no such gate can work: for a single atomic fact, a good
    # question IS the fact minus its answer, put as a question. That is what
    # asking about a fact means. `Wie hoch war die Arbeitslosenquote im August
    # 2025?` and `Was schüren geopolitische Konflikte?` have the same shape and
    # the same overlap with their facts; what separates them is whether the
    # answer is determinate, and that is not lexical either - a units rule
    # refused 7 of 15 accepted answers, `knapp ein Fünftel` among them.
    #
    # Whether a question is worth asking is a judgement, so the verifier makes
    # it. `not_recoverable` already carries it: it cannot get a determinate
    # answer back out of the passages for a vague question, which is the same
    # thing measured where it can be measured. The exact-match check above
    # stays, because a fact handed back with a question mark is not a
    # judgement call.

    # Read on the question and its answer together, and only believed when it
    # answers: lingua returns nothing below forty characters of prose, and
    # most questions are shorter than that. A detector that cannot tell is
    # not evidence of the wrong language.
    written = detect(f"{asked} {target_answer or ''}".strip())
    if language and written and written != language:
        return (
            QuestionRejection.MALFORMED,
            f"it is written in {written} and its facts are in {language}",
        )
    return None


#: How many things a question has to be about before it is taken to name
#: something, when it names no name and no number. Measured over twelve
#: questions: the ones nobody could have asked cold carry one or two content
#: words - `What specific components are included?`, `Which criteria are
#: used?` - and the ones a person would type carry four or five.
_NAMES_SOMETHING = 3


def anchored(question: str, language: str | None) -> bool:
    """Whether a question names anything on its own, read off its parse.

    The measurement behind an opinion. `unanchored` is the verifier's
    judgement and the verifier is wrong about it often enough to matter: over
    one topic it rejected six questions naming a party, a period and a duty
    apiece, four of them unanswerable ones where a model that could not find
    the answer reports that the question named nothing. Asking it to copy the
    subject out instead fixed most of that and not all.

    So the judgement may only reject a question this also calls thin. A
    question naming a name or a number names something specific; so does one
    about three or more things. `What specific components are included?` is
    about two and names neither.
    """
    return bool(claim(question, language).units) or (
        len(content(question, language)) >= _NAMES_SOMETHING
    )


def near_verdict(
    near: Neighbour | None, *, answerable: bool, threshold: float
) -> tuple[str, str] | None:
    """What the nearest accepted question says about this one.

    One probe answering two gates. A near twin of an accepted question is a
    duplicate and adds nothing to the benchmark; a near twin of an accepted
    *answerable* question, when this one is meant to be unanswerable, is a
    question the corpus demonstrably does answer.
    """
    if near is None or near.similarity < threshold:
        return None
    if near.answerable and not answerable:
        return (
            QuestionRejection.ANSWERABLE_AFTER_ALL,
            (
                f"the corpus answers this: it is {near.similarity:.2f} from the "
                f"accepted question {near.question_text!r}"
            ),
        )
    return (
        QuestionRejection.DUPLICATE,
        f"{near.similarity:.2f} from the accepted question {near.question_text!r}",
    )


@dataclass(frozen=True)
class Reading:
    """What the verifier made of one question and its passages.

    `recovered` is the answer it got back out of them, or None for NOT IN
    PASSAGE. `stands_alone` is whether the question names a subject at all.
    `names_its_source` is whether it says which material holds the answer,
    which is the opposite failure and was for a while mistaken for a virtue.
    """

    recovered: str | None
    stands_alone: bool
    names_its_source: bool = False


class Verifier:
    """A second model, asked to get the answer back out of the passages."""

    def __init__(self, client: Client) -> None:
        """Initialises the verifier with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model doing the verifying."""
        return self._client.model

    def read(
        self,
        question: str,
        passages: Sequence[str],
        thread: Sequence[tuple[str, str | None]] = (),
    ) -> Reading:
        """Answers one question from these passages, and judges the question.

        Two verdicts out of one call. The phrasing judgement is a model's to
        make and a structural check cannot make it, and asking for it here
        costs nothing: the call is happening anyway and the model is already
        looking at both the question and the material.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        numbered = "\n\n".join(
            f"[{position}] {passage}" for position, passage in enumerate(passages, 1)
        )
        # A follow-up is asked in a conversation, so it is judged in one. Read
        # alone, `And for an urgent one?` has no answer in any passage.
        asked = (
            "\n".join(f"Q: {q}\nA: {a or 'not in the material'}" for q, a in thread)
            + f"\nQ: {question}"
            if thread
            else f"Question: {question}"
        )
        got = self._client.answer(
            system=_VERIFY,
            user=f"Passages:\n{numbered}\n\n{asked}",
            shape=_Recovered,
        )
        return Reading(
            recovered=(
                got.answer.strip() if got.in_passage and got.answer.strip() else None
            ),
            # Whether it names a subject, read off the subject it named. Asked
            # as a yes or no it was answered `no` for questions naming a party,
            # a period and a duty apiece - six of ten rejections in one topic,
            # four of them on unanswerable questions, where a model that could
            # not find the answer says the question named nothing. Copying a
            # span out of the question is a task a small model does not
            # confuse with reading the passages.
            stands_alone=bool(got.subject.strip()),
            names_its_source=got.names_its_source,
        )


def agrees(
    recovered: str,
    target: str,
    language: str | None,
    form: str = AnswerForm.VALUE,
    overlap: float = OVERLAP,
    question: str = "",
) -> bool:
    """Whether what the verifier got back says what the target answer says.

    Two tests, both of which have to pass.

    How the second is applied depends on the FORM the answer was asked to
    take. A value is compared whole, because every word of one is the answer.
    A list or an explanation is compared by how much of it came back: `it is
    raised through the web form, and confirmed by email before work begins`
    recovered as `raised through the web form and confirmed by email` is the
    same answer, and asking for every lemma of prose to survive a paraphrase
    refuses answers the verifier found. QUESTIONS_ANSWER_OVERLAP is where the
    share is set.

    Every number and name the target asserts has to occur in what came back.
    That is extraction's `unsupported_addition` check pointed the other way,
    and it is what stops `4 hours` passing for `48 hours` - no measure of
    likeness would, because the two are as alike as two answers get.

    And what the target is ABOUT has to occur in what came back, compared on
    content lemmas. That half replaced string containment, which was refusing
    answers the verifier had found: `geopolitische Umbrüchen und
    fortschreitender Digitalisierung` is the same answer as `geopolitische
    Umbrüche und fortschreitende Digitalisierung` and shares not one inflected
    form with it. Measured over the twelve pairs one run produced, containment
    agreed with a person on 8 and lemmas on 11, with nothing newly accepted
    that a person called different.

    What it still misses is German inflection the pipeline gets wrong:
    `de_core_news_md` lemmatises `Umbrüche` to `Umbruch` and leaves
    `Umbrüchen` alone, so those two do not meet, and the same happens to
    `einfache` against `einfacheren`.

    Embedding the two answers and comparing them closes both, and was
    measured rather than assumed: over fourteen pairs, lemmas got 12 right
    with two such refusals, and the numbers check plus a cosine of 0.92 got
    13 with one wrong acceptance. The ranges overlap - the same answers bottom
    out at 0.920 and different ones reach 0.942 - so no threshold separates
    them, and `4 hours` against `48 hours` is 0.942.

    Lemmas are kept because the errors are not equal. A refusal loses a good
    question, which is costly and safe. An acceptance puts a question in the
    benchmark whose answer the material does not give, which marks a correct
    chatbot wrong - the failure this gate exists for.

    A target with nothing to compare either way - a bare `yes` - falls back to
    containment, which is all that is left.
    """
    # Numbers compared with their separators folded away. This corpus is
    # German and writes `0,3`; the verifier answers `0.3%` as often as not,
    # and the two are one number written in two locales. Folding keeps the
    # comparison that matters - `4` is still not `48` - and drops the one
    # that never did.
    found = {_folded(one) for one in vocabulary(recovered, language)}
    units = claim(target, language).units
    if units and not all(_folded(unit) in found for unit in units):
        return False

    wanted = content(target, language)
    if wanted:
        got = content(recovered, language)
        if form == AnswerForm.VALUE:
            return wanted <= got
        # Measured over what the ANSWER adds, not over the question restated
        # inside it. A model asked for prose writes `Die Regionen, in die
        # chinesische Produkte exportiert werden, umfassen Südostasien und
        # Afrika`; the verifier answers `Südostasien, Afrika und Europa`. The
        # two agree about everything the answer carries and share two lemmas
        # of seven, so counting the question's own words rejected it.
        answered = wanted - content(question, language) if question else wanted
        if not answered:
            return bool(wanted & got)
        return len(answered & got) / len(answered) >= overlap

    left, right = normalised(recovered), normalised(target)
    return bool(left) and (left in right or right in left)


class QuestionChecker:
    """Puts one candidate through every gate, cheapest first."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        verifier: Verifier,
        nearest,
        threshold: float,
        judge_phrasing: bool = True,
        bounds: Mapping[str, tuple[int, int]] | None = None,
        overlap: float = OVERLAP,
        long_answer_chars: int = LONG_ANSWER_CHARS,
    ) -> None:
        """Initialises the checker with its collaborators.

        `nearest` is a callable rather than the catalogue itself: what this
        needs of the database is one probe, and taking the whole repository
        for it would let a gate read anything.

        `judge_phrasing` is whether the verifier's opinion of the question
        may reject one. Those gates are the only ones here that are an
        opinion rather than a measurement, and an opinion needs an
        independent holder: a writer marking its own work rejected `According
        to the ECB and NCAs, who conducts the due diligence check?` for
        naming nothing. The factory turns it off when QUESTIONS_VERIFIER_MODEL
        is unset, and the verdict is logged either way. Recoverability stays
        on regardless, because that one is checkable against the passage.
        """
        self._embedder = embedder
        self._verifier = verifier
        self._nearest = nearest
        self._judge_phrasing = judge_phrasing
        self._bounds = dict(bounds or BOUNDS)
        self._overlap = overlap
        self._long_answer_chars = long_answer_chars
        self._threshold = threshold

    def check(
        self, candidate: Candidate, seen: Sequence[CheckedQuestion] = ()
    ) -> CheckedQuestion:
        """Judges one candidate, returning it stored either way.

        `seen` is what this run has already accepted. Without it the dedup
        gate would compare a candidate only against what was in the database
        when the run started, and a topic would happily write the same
        question four times.

        Raises:
            ModelUnavailable: If the verifier could not be reached.
        """
        failed = structural(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            language=candidate.group.language,
            statements=candidate.group.statements,
            form=candidate.spec.form,
            bounds=self._bounds,
            titles=candidate.group.titles,
        )
        if failed:
            return self._verdict(candidate, failed, None, self._long_answer_chars)

        embedding = self._embedder.embed(candidate.question_text)
        failed = near_verdict(
            self._near(embedding, seen),
            answerable=candidate.answerable,
            threshold=self._threshold,
        )
        if failed:
            return self._verdict(candidate, failed, embedding, self._long_answer_chars)

        return self._verdict(
            candidate,
            self._round_trip(candidate),
            embedding,
            self._long_answer_chars,
        )

    def _round_trip(self, candidate: Candidate) -> tuple[str, str] | None:
        """Asks whether the passages give the answer back, and how it reads."""
        read = self._verifier.read(
            candidate.question_text,
            candidate.group.passages,
            candidate.thread,
        )

        # Judged before the answer is, and for both kinds of question: an
        # unanswerable question nobody would ask is as useless as an
        # answerable one, and a chatbot declining it proves nothing.
        #
        # Never for a follow-up. `And for an urgent one?` names nothing and
        # is exactly the question a person asks second; leaning on the thread
        # is what a follow-up is for, so judging it as though it had been
        # asked cold would reject every one of them.
        # The opposite failure to the one below, and the one that matters
        # more: a question carrying its own source has already done the work
        # it was meant to test. Judged for a follow-up too - leaning on the
        # conversation is allowed, naming the file is not.
        if read.names_its_source:
            leak = (
                QuestionRejection.LEAKS_SOURCE,
                (
                    "the verifier says it names the material the answer is in, so "
                    "it asks a question and answers half of it"
                ),
            )
            if self._judge_phrasing:
                return leak
            log.info(
                "keeping %r: %s, but the writer judged itself",
                candidate.question_text,
                leak[1],
            )

        # The verifier's opinion, and only where the question is thin enough
        # for it to be worth holding. A question naming a name, a number or
        # three things was asked about something, whatever the verifier made
        # of it.
        if (
            not read.stands_alone
            and not candidate.follows
            and not anchored(candidate.question_text, candidate.group.language)
        ):
            reason = (
                "the verifier says it names nothing a person searching would "
                "know, and it names no name, no number and fewer than "
                f"{_NAMES_SOMETHING} things, so it could not have been asked "
                "without the passage"
            )
            if self._judge_phrasing:
                return QuestionRejection.UNANCHORED, reason
            # Logged and kept. An opinion needs an independent holder, and
            # there is none when the writer is marking its own work.
            log.info(
                "keeping %r: %s, but the writer judged itself",
                candidate.question_text,
                reason,
            )

        if candidate.answerable:
            target = candidate.target_answer or ""
            if read.recovered is None:
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    "the verifier could not find the answer in the cited passages",
                )
            if not agrees(
                read.recovered,
                target,
                candidate.group.language,
                candidate.spec.form,
                self._overlap,
                candidate.question_text,
            ):
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    (
                        f"the verifier recovered {read.recovered!r} where the "
                        f"target answer is {target!r}"
                    ),
                )
            return None

        if read.recovered is not None:
            return (
                QuestionRejection.ANSWERABLE_AFTER_ALL,
                f"the cited passages answer it after all, with {read.recovered!r}",
            )
        return None

    def _near(
        self, embedding: list[float], seen: Sequence[CheckedQuestion]
    ) -> Neighbour | None:
        """The nearest accepted question, stored or written earlier in this run."""
        near = self._nearest(embedding)
        for other in seen:
            if other.embedding is None:
                continue
            score = cosine(embedding, other.embedding)
            if near is None or score > near.similarity:
                near = Neighbour(other.question_text, other.answerable, score)
        return near

    @staticmethod
    def _verdict(
        candidate: Candidate,
        failed: tuple[str, str] | None,
        embedding: list[float] | None,
        long_answer: int = LONG_ANSWER_CHARS,
    ) -> CheckedQuestion:
        """Assembles one judged question, kept whichever way it went."""
        code, reason = failed if failed else (None, "")
        if code:
            log.info("rejected %r: %s (%s)", candidate.question_text, reason, code)
        return CheckedQuestion(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            criteria=candidate.group.criteria(
                len(candidate.target_answer) if candidate.target_answer else None,
                follows=candidate.follows,
                long_answer=long_answer,
            ),
            language=candidate.group.language,
            status=QuestionStatus.REJECTED if code else QuestionStatus.ACCEPTED,
            rejected_reason=code,
            fact_ids=tuple(fact.id for fact in candidate.group.facts),
            embedding=embedding,
            thread_position=candidate.thread_position,
            question_type=candidate.spec.name,
            # An unanswerable question has no answer, so it has no form: the
            # column says what shape the answer takes and there is none.
            answer_form=candidate.spec.form if candidate.answerable else None,
            planned_difficulty=candidate.planned_difficulty,
        )
