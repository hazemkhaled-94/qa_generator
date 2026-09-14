"""The gates a written question has to pass.

Four of them, applied cheapest first, because each one that fires saves the
cost of the ones behind it:

  malformed     structural, free
  duplicate     one index probe against the questions already accepted
  answerable    the same probe, when an unanswerable question has a twin
                the corpus does answer
  recoverable   one call to a second model, given only the cited passages

The last is the one no similarity measure makes. It asks whether the answer
can be got back out of the passages, which is what a question is for; a
paraphrase, a decomposition and a resolved pronoun all survive it, and a
question whose answer is nowhere in its evidence does not, however much it
looks like one that is.

Two things about it are load-bearing. The verifier is a different model from
the writer, because a model marking its own work agrees with itself. And it
sees only the cited passages, never the corpus: what is being measured is the
dataset, not a retriever.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from database.qa_generator import QuestionRejection, QuestionStatus
from llm.client import Client
from nlp.analysis import claim, normalised, vocabulary
from nlp.language import detect
from question_generation.embedding import Embedder, cosine
from question_generation.models import Candidate, CheckedQuestion, Neighbour

log = logging.getLogger(__name__)

_VERIFY = """You answer a question using ONLY the passages you are given.

You have no other knowledge. Nothing you know from anywhere else counts, and
nothing you can work out counts either: if the passages do not state it, it is
not there.

Answer in two parts:
- `in_passage` is true ONLY when the passages state the answer outright. If
  you are completing, inferring, rounding or assuming, it is false.
- `answer` is the answer, taken from the passages, when in_passage is true.
  Leave it empty otherwise.

Saying it is not in the passages is a correct answer and is the one we are
looking for whenever it is true. Guessing is the failure.
"""


class _Recovered(BaseModel):
    """What the verifier got back out of the passages."""

    in_passage: bool = Field(
        description="True only when the passages state the answer outright. "
        "False is NOT IN PASSAGE, and is a correct answer."
    )
    answer: str = Field(
        default="",
        description="The answer, taken from the passages. Empty when "
        "in_passage is false.",
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

    Returns the code to store and the reason to log: five ways of being
    malformed share one code, because none of them measures anything a
    report would group on, and the reason belongs where a person debugging
    reads it.
    """
    asked = question_text.strip()
    if not asked:
        return QuestionRejection.MALFORMED, "the model returned no question"
    if not asked.endswith("?"):
        return QuestionRejection.MALFORMED, "it does not end in a question mark"
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


class Verifier:
    """A second model, asked to get the answer back out of the passages."""

    def __init__(self, client: Client) -> None:
        """Initialises the verifier with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model doing the verifying."""
        return self._client.model

    def recover(self, question: str, passages: Sequence[str]) -> str | None:
        """Answers one question from these passages, or None if they do not.

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
        return got.answer.strip() if got.in_passage and got.answer.strip() else None


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
            statements=[fact.statement for fact in candidate.group.facts],
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
        """Asks whether the cited passages give the answer back."""
        recovered = self._verifier.recover(
            candidate.question_text, candidate.group.passages
        )
        if candidate.answerable:
            target = candidate.target_answer or ""
            if recovered is None:
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    "the verifier could not find the answer in the cited passages",
                )
            if not agrees(recovered, target, candidate.group.language):
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    (
                        f"the verifier recovered {recovered!r} where the target "
                        f"answer is {target!r}"
                    ),
                )
            return None

        if recovered is not None:
            return (
                QuestionRejection.ANSWERABLE_AFTER_ALL,
                f"the cited passages answer it after all, with {recovered!r}",
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
