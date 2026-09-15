"""The gates a written question has to pass.

Applied cheapest first, because each one that fires saves the cost of those
behind it:

  malformed       structural, free
  restates_fact   structural, free: the question is about what its fact is
                  about and nothing more
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

`unanchored` rides on the same call. Whether a question reads like one a
person would type is a judgement, not a measurement, and no structural check
makes it - but a model already looking at the question and the material can,
and asking costs nothing extra. It is the half of quality `restates_fact`
cannot reach: a question can add a word its fact lacks and still be one
nobody would ask.

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
from question_generation.models import Candidate, CheckedQuestion, Neighbour

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

SECOND, judge the question itself, as a question. Imagine somebody who has
never seen these passages and is searching a corpus of thousands of pages.

- `stands_alone` is true only if that person could have typed this question
  and would know what it is about. It names its subject: the institution, the
  document, the rule, the period, the thing being regulated.

  It is FALSE when the question only makes sense with the passage in front of
  you - when it opens with a bare "What" or "Which" and names nothing, or
  refers to "the requirements", "the report", "this circular" without saying
  which, or reads like a sentence from the passage with one part removed.

Judge these two independently. A question can be perfectly answerable by the
passages and still be one nobody would ever ask.
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
    stands_alone: bool = Field(
        default=True,
        description="True only if somebody who has never read the passages "
        "could have typed this question and would know what it is about. "
        "False if it names nothing, refers to `the requirements` or `this "
        "circular` without saying which, or is a passage sentence with one "
        "part removed.",
    )


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
) -> tuple[str, str] | None:
    """The gates that need neither a model nor an index, or None if it passes.

    Returns the code to store and the reason to log. The several ways of
    being malformed share one code, because none of them measures anything a
    report would group on and the reason belongs where a person debugging
    reads it. Restating the fact gets a code of its own: it is how a writer
    fails when the prompt is right and the model ignores it, which is worth
    counting apart from a missing question mark.
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
    answer = (target_answer or "").strip()
    if answer and claim(answer, language).verbs:
        return (
            QuestionRejection.MALFORMED,
            (
                f"the target answer {answer!r} describes an action rather than "
                f"naming a thing; what can be scored is a value, an amount, a "
                f"date, a name or a duty"
            ),
        )

    # The defect this stage was first shipped with, made measurable. A
    # question built by taking its fact and replacing one part with a
    # question word is about exactly what the fact is about and nothing more:
    # `Geopolitische Konflikte schüren Unsicherheit` becomes `Was schüren
    # geopolitische Konflikte?`, which nobody searching a corpus would type.
    #
    # Compared on content lemmas - what each is about - rather than on words,
    # so an inflection or a rephrasing is not mistaken for added meaning. A
    # question that names its document, its institution or its year adds a
    # lemma the fact does not have and passes; one that permutes the fact has
    # nothing of its own to show.
    if statements and answerable:
        asking = content(asked, language)
        told = set().union(*(content(one, language) for one in statements))
        if asking and not asking - told:
            return (
                QuestionRejection.RESTATES_FACT,
                (
                    f"it is about {', '.join(sorted(asking))} and so is its fact, "
                    f"so it adds nothing a person searching for this would have "
                    f"to know to ask it"
                ),
            )

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
    PASSAGE. `stands_alone` is whether somebody who has not read them could
    have asked the question at all.
    """

    recovered: str | None
    stands_alone: bool


class Verifier:
    """A second model, asked to get the answer back out of the passages."""

    def __init__(self, client: Client) -> None:
        """Initialises the verifier with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model doing the verifying."""
        return self._client.model

    def read(self, question: str, passages: Sequence[str]) -> Reading:
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
        got = self._client.answer(
            system=_VERIFY,
            user=f"Passages:\n{numbered}\n\nQuestion: {question}",
            shape=_Recovered,
        )
        return Reading(
            recovered=(
                got.answer.strip() if got.in_passage and got.answer.strip() else None
            ),
            stands_alone=got.stands_alone,
        )


def agrees(recovered: str, target: str, language: str | None) -> bool:
    """Whether what the verifier got back says what the target answer says.

    The same test extraction uses for `unsupported_addition`, pointed the
    other way: every number, name and date the target asserts has to occur in
    what was recovered. Comparing the strings would fail on a verifier that
    wrote `48 hours` where the target says `within 48 hours`, and an
    embedding would pass one that wrote `4 hours`.

    A target carrying no such unit - a duty, a yes - has nothing to check
    that way, so those fall back to containment either way round.
    """
    units = claim(target, language).units
    if units:
        found = vocabulary(recovered, language)
        return all(unit in found for unit in units)
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
    ) -> None:
        """Initialises the checker with its collaborators.

        `nearest` is a callable rather than the catalogue itself: what this
        needs of the database is one probe, and taking the whole repository
        for it would let a gate read anything.
        """
        self._embedder = embedder
        self._verifier = verifier
        self._nearest = nearest
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
        )
        if failed:
            return self._verdict(candidate, failed, None)

        embedding = self._embedder.embed(candidate.question_text)
        failed = near_verdict(
            self._near(embedding, seen),
            answerable=candidate.answerable,
            threshold=self._threshold,
        )
        if failed:
            return self._verdict(candidate, failed, embedding)

        return self._verdict(candidate, self._round_trip(candidate), embedding)

    def _round_trip(self, candidate: Candidate) -> tuple[str, str] | None:
        """Asks whether the passages give the answer back, and how it reads."""
        read = self._verifier.read(candidate.question_text, candidate.group.passages)

        # Judged before the answer is, and for both kinds of question: an
        # unanswerable question nobody would ask is as useless as an
        # answerable one, and a chatbot declining it proves nothing.
        if not read.stands_alone:
            return (
                QuestionRejection.UNANCHORED,
                (
                    "the verifier could not have asked this without the passage in "
                    "front of it: it names nothing a person searching would know"
                ),
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
    ) -> CheckedQuestion:
        """Assembles one judged question, kept whichever way it went."""
        code, reason = failed if failed else (None, "")
        if code:
            log.info("rejected %r: %s (%s)", candidate.question_text, reason, code)
        return CheckedQuestion(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            difficulty=candidate.group.difficulty,
            language=candidate.group.language,
            status=QuestionStatus.REJECTED if code else QuestionStatus.ACCEPTED,
            rejected_reason=code,
            fact_ids=tuple(fact.id for fact in candidate.group.facts),
            embedding=embedding,
        )
