"""The gates a written question has to pass.

Applied cheapest first, because each one that fires saves the cost of those
behind it:

  malformed       structural, free
  answer_too_short  free: nothing worth scoring against
  leaks_source    free where the title is quoted, the verifier's otherwise
  duplicate       one index probe against the questions already accepted
  answerable      the same probe, when an unanswerable question has a twin
                  the corpus does answer
  unanchored      a question nobody could have asked without the passage
  recoverable     the answer is not in the evidence the question cites

The last three come out of one model call. Recoverability is the one no
similarity measure makes: it asks whether the answer can be got back out of
the passages, which is what a question is for, and a paraphrase, a
decomposition and a resolved pronoun all survive it where a threshold would
not.

`unanchored` and `leaks_source` ride on the same call. Whether a question
names its subject, and whether it gives away which document answers it, are
judgements rather than measurements; no structural check makes either, but a
model already looking at the question and the material can, and asking costs
nothing extra.

Two things are load-bearing. The verifier is a different model from the
writer, because a model marking its own work agrees with itself. And it sees
only the cited passages, never the corpus: what is being measured is the
dataset, not a retriever.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from database.qa_generator import QuestionRejection, QuestionStatus
from llm.client import Client
from nlp.analysis import claim, content, normalised, vocabulary
from nlp.language import detect
from question_generation.embedding import Embedder, cosine
from question_generation.models import (
    LONG_ANSWER_CHARS,
    Candidate,
    CheckedQuestion,
    Neighbour,
)

log = logging.getLogger(__name__)

_VERIFY = """You check one test question for a document-search chatbot.

You are given some PASSAGES and a QUESTION. Do two separate things.

FIRST, answer the question using ONLY those passages. You have no other
knowledge. Nothing you know from anywhere else counts, and nothing you can
work out counts either: if the passages do not state it, it is not there.

- `in_passage` is true ONLY when the passages state the answer outright. If
  you are completing, inferring, rounding or assuming, it is false.
- `answer` is the answer, taken from the passages, when in_passage is true.
  Leave it empty otherwise.

Saying it is not in the passages is a correct answer and is the one we are
looking for whenever it is true. Guessing is the failure.

SECOND, answer two narrow questions about the question itself. They are about
different things and are easy to confuse, so read both.

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

- `stands_alone`: could somebody who has never read the passage tell what is
  being asked? Is there a subject in the question at all?

  TRUE:  "What fee applies to a banking licence application?"
  FALSE: "What specific components are included?"      (nothing named)
  FALSE: "For which models do the requirements apply?" (which requirements?)

  Judge only that. Do not mark it false for being broad, for being easy, for
  being oddly worded, or for being one you would not have asked. And do not
  mark it false for failing to name a document - a question is SUPPOSED not
  to name one.

All three judgements are independent. A question can be answerable by the
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
        description="The answer, taken from the passages. Empty when "
        "in_passage is false.",
    )
    names_its_source: bool = Field(
        default=False,
        description="True if the question says WHERE the answer is: naming or "
        "quoting a document, report, circular, named regulation, section or "
        "heading. Naming a party, duty, period or regulated thing is not "
        "naming a source.",
    )
    stands_alone: bool = Field(
        default=True,
        description="True if somebody who has never read the passages could "
        "tell what is being asked. False only if the question names no "
        "subject at all. Not false for failing to name a document: a question "
        "is supposed not to name one.",
    )


#: The shortest document title worth matching a question against. Below it a
#: title is a word rather than a name - this corpus has `Contents` - and a
#: question may contain one without citing anything.
_TITLE_CHARS = 12


def _bare(text: str) -> str:
    """Folds a string for comparison, sentence punctuation and all."""
    return normalised(text).rstrip("?.!")


def structural(
    *,
    question_text: str,
    target_answer: str | None,
    answerable: bool,
    language: str | None,
    statements: Sequence[str] = (),
    min_answer_chars: int = 0,
    titles: Sequence[str] = (),
) -> tuple[str, str] | None:
    """The gates that need neither a model nor an index, or None if it passes.

    Returns the code to store and the reason to log. The several ways of
    being malformed share one code, because none of them measures anything a
    report would group on and the reason belongs where a person debugging
    reads it. The two that do get a code of their own - an answer too short
    to score, and a question quoting the title of its own source - are the
    two a report should be able to count.
    """
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

    # An answer carrying a verb is a fragment of the sentence rather than the
    # thing asked for: `Verwarnungen aussprechen`, `nachvollziehbar zu
    # begründen`, `nahmen Produkte vom Markt`. Any verb and not only a finite
    # one, because two of those three are infinitives and make no claim - a
    # phrase can describe an action without asserting it.
    #
    # Nothing narrower is checked here. `48 hours` answering `within how many
    # hours ...` shares its head noun with the question and is a perfectly
    # ordinary pair, so overlap between the two is not evidence of anything.
    # A floor on the answer, so a question is not scored against something
    # too thin to be worth scoring: `7`, `8%`, `Nein`. It is a blunt
    # instrument and QUESTIONS_MIN_ANSWER_CHARS is where it is set, because
    # the cost is real - on one corpus a floor of 15 refused 41% of the
    # accepted answers, `70%` and `2025` among them. Raise it to force
    # substance, lower it to keep bare values.
    answer = (target_answer or "").strip()
    if answer and len(answer) < min_answer_chars:
        return (
            QuestionRejection.ANSWER_TOO_SHORT,
            (
                f"the target answer {answer!r} is {len(answer)} characters and "
                f"QUESTIONS_MIN_ANSWER_CHARS is {min_answer_chars}"
            ),
        )

    # Only for an answer of more than one word. `Verwarnungen aussprechen`,
    # `nachvollziehbar zu begründen` and `nahmen Produkte vom Markt` - the
    # three real failures - are all several. A single word is a thing by
    # construction, and `de_core_news_md` tags `dreihundert` as a verb, so
    # checking one refuses a good answer to catch a rare bad one. Losing
    # data is the worse error, as with the phrasing gate.
    if answer and len(answer.split()) > 1 and claim(answer, language).verbs:
        return (
            QuestionRejection.MALFORMED,
            (
                f"the target answer {answer!r} describes an action rather than "
                f"naming a thing; what can be scored is a value, an amount, a "
                f"date, a name or a duty"
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
            stands_alone=got.stands_alone,
            names_its_source=got.names_its_source,
        )


def agrees(recovered: str, target: str, language: str | None) -> bool:
    """Whether what the verifier got back says what the target answer says.

    Two tests, both of which have to pass.

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

    The one it still misses is morphology the pipeline gets wrong:
    `de_core_news_md` lemmatises `Umbrüche` to `Umbruch` and leaves
    `Umbrüchen` alone, so the two do not meet. Stemming would close it and is
    not worth the false accepts it would open on twelve pairs of evidence.

    A target with nothing to compare either way - a bare `yes` - falls back to
    containment, which is all that is left.
    """
    units = claim(target, language).units
    if units and not all(unit in vocabulary(recovered, language) for unit in units):
        return False

    wanted = content(target, language)
    if wanted:
        return wanted <= content(recovered, language)

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
        min_answer_chars: int = 0,
        long_answer_chars: int = LONG_ANSWER_CHARS,
    ) -> None:
        """Initialises the checker with its collaborators.

        `nearest` is a callable rather than the catalogue itself: what this
        needs of the database is one probe, and taking the whole repository
        for it would let a gate read anything.

        `judge_phrasing` is whether the verifier's opinion of the question
        may reject one. It is the only gate here that is an opinion rather
        than a measurement, and an opinion needs an independent holder: a
        writer marking its own work rejected `According to the ECB and NCAs,
        who conducts the due diligence check?` for naming nothing. The
        factory turns it off when QUESTIONS_VERIFIER_MODEL is unset, and the
        verdict is logged either way. Recoverability stays on regardless,
        because that one is checkable against the passage.
        """
        self._embedder = embedder
        self._verifier = verifier
        self._nearest = nearest
        self._judge_phrasing = judge_phrasing
        self._min_answer_chars = min_answer_chars
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
            min_answer_chars=self._min_answer_chars,
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
            candidate.question_text, candidate.group.passages, candidate.thread
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

        if not read.stands_alone and not candidate.follows:
            reason = (
                "the verifier says it names nothing a person searching would "
                "know, so it could not have been asked without the passage"
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
            if not agrees(read.recovered, target, candidate.group.language):
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
        )
