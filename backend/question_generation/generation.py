"""Writing one question from a sample of facts, to the plan that asked for it.

Three kinds of call, one rendering. A question the facts answer, a question
they deliberately do not, and the question somebody would ask next. Each is
one call against one sample with one shape back.

What each asks for comes from the plan's type, in `types.py`. Nothing about a
domain is written here: a type says what kind of thing to ask for, and the
same eleven types are askable of a manual, a contract or a report.

The writer is shown the passage each fact came from, and says which facts its
question needs. Neither was true of the first version of this module, and both
failures had the same cause: a single atomic statement is one triple, so the
only question that can be built out of it is that statement with one part
replaced by a question word. The passage is what a question can be phrased
from; the fact is what it has to be answered by.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.client import Client
from question_generation.models import Candidate, FactGroup
from question_generation.planning import Plan
from question_generation.types import PROMPT_VERSION

__all__ = ["PROMPT_VERSION", "QuestionWriter"]

_PERTURB = """You write one test question that the material does NOT answer.

These measure whether a chatbot says it does not know instead of inventing an
answer. So the question has to be a plausible thing to ask of this material
and have no answer anywhere in it.

You are given one FACT and the PASSAGE it came from. Take the fact and move it
just out of reach. Make ONE change:
- ask about a neighbouring thing the material does not cover,
- ask for a detail of the same subject it does not state, or
- ask about a different party, place, period or category.

Rules, all of them mandatory:
- ONE question, ending in a question mark.
- NAME THE SUBJECT, NEVER THE SOURCE. Say what you are asking about - the
  party, the thing, the period - so the question reads as though it belongs to
  this material, because a question about an unrelated subject tests nothing:
  any chatbot declines that one. But NEVER say which document, report or
  section it would be in.
- It must NOT be answerable from the fact or the passage. If reading either
  answers your question, you have written the wrong question.
- Do not ask something absurd and do not invent a thing that does not exist.
  Both are too easy to decline.
- Write in the language of the fact.

Worked example:

  FACT: A standard support request is answered within 48 hours.
  PASSAGE: Standard requests are answered within 48 hours on working days.
  These times are set out in the service agreement.

  WRONG  "How long does a standard support request take to answer?"
         (that is the fact itself, and it is answered)
  WRONG  "What is the top speed of a swallow?"
         (nothing to do with this material; declining it tests nothing)
  WRONG  "What does the service agreement say about public holidays?"
         (names its source, which no asker would know to do)

  RIGHT  "How long is allowed for answering a standard support request raised
          on a public holiday?"
         (same subject, same shape, a condition the material does not cover)
"""

_FOLLOW = """You write the question somebody would ask NEXT.

You are given the FACTS, the PASSAGE they came from, and the CONVERSATION so
far: one or more questions already asked and the answers they got.

Write the next question in that conversation. Somebody has just been told the
last answer and wants to know one more thing about the same material.

This one is different from a question asked cold, and the difference is the
point: it MAY rely on the conversation. "And for urgent requests?" is a
perfectly good follow-up. You do not have to name the subject again, because
the person you are talking to already knows it.

Rules, all of them mandatory:
- ONE question, ending in a question mark.
- It must be answered by the FACTS, like any other. A follow-up whose answer
  is not in the material tests nothing.
- NEVER SAY WHERE THE ANSWER IS. No naming or quoting a document, a report, a
  section or a heading, here as anywhere else.
- DO NOT REPEAT a question already in the conversation, and do not ask one
  the last answer already gave. It has to want something new.
- Write in the language of the facts.
- `facts` is the NUMBERS of the facts your question needs.
"""


class _Answered(BaseModel):
    """A question the facts answer, as the model is asked to return it."""

    question: str = Field(
        description="ONE question, ending in a question mark, naming what it "
        "asks about the way a searcher would have to."
    )
    answer: str = Field(description="The answer, in the form the instructions ask for.")
    facts: list[int] = Field(
        default_factory=list,
        description="The NUMBERS of the facts this question needs.",
    )


class _Unanswered(BaseModel):
    """A question the material does not answer."""

    question: str = Field(
        description="ONE question, ending in a question mark, that reads as if "
        "it belongs to this material and that it does not answer."
    )


class QuestionWriter:
    """Writes one question per sample of facts with a served model."""

    def __init__(self, client: Client) -> None:
        """Initialises the writer with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model writing the questions."""
        return self._client.model

    def write(self, sample: FactGroup, plan: Plan) -> Candidate:
        """Asks the model for one question of the kind the plan wants.

        Returns whatever came back, unjudged. An empty question is a
        candidate like any other: the gates reject it and the rejection is
        what the drop rate is measured from.

        The returned candidate carries the facts the model said it used, not
        the whole sample. A sample is an offer; a question that needed one
        fact of three is a single-passage question, and recording it as
        anything else is a difficulty nobody can reproduce.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        if not plan.answerable:
            return self._unanswerable(sample, plan)

        written = self._client.answer(
            system=plan.spec.system(spans=plan.spans),
            user=self._prompt(sample),
            shape=_Answered,
        )
        return self._candidate(
            plan,
            question=written.question,
            answer=written.answer,
            group=self._used(sample, written.facts),
        )

    def follow_up(
        self,
        sample: FactGroup,
        thread: tuple[tuple[str, str | None], ...],
        plan: Plan,
    ) -> Candidate:
        """Asks the model for the question somebody would ask next.

        Written from the same sample as the thread it joins, so a follow-up
        is about the same material rather than a fresh question that happens
        to come after one. Its own type, so a thread can move from a value to
        the reason behind it.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        written = self._client.answer(
            system=(
                f"{_FOLLOW}\nWHAT TO ASK NEXT: {plan.spec.asks}\n\n"
                f"THE ANSWER IS {plan.spec.answer_rule}"
            ),
            user=f"{self._prompt(sample)}\n\n{self._conversation(thread)}",
            shape=_Answered,
        )
        return self._candidate(
            plan,
            question=written.question,
            answer=written.answer,
            group=self._used(sample, written.facts),
            thread=thread,
        )

    def _unanswerable(self, sample: FactGroup, plan: Plan) -> Candidate:
        """Moves one fact out of reach and asks about where it went.

        From one fact and not from the sample: moving a claim just out of
        reach is a change to one claim, and a reader given two would have two
        ways to notice.
        """
        first = FactGroup(sample.facts[:1])
        written = self._client.answer(
            system=f"{_PERTURB}\nTHE KIND OF QUESTION TO ASK: {plan.spec.asks}",
            user=self._prompt(first),
            shape=_Unanswered,
        )
        return Candidate(
            question_text=written.question.strip(),
            # An unanswerable question is scored on behaviour and not on
            # content, which the questions table enforces with a CHECK.
            target_answer=None,
            answerable=False,
            group=first,
            spec=plan.spec,
            planned_difficulty=plan.band,
        )

    @staticmethod
    def _candidate(
        plan: Plan,
        *,
        question: str,
        answer: str,
        group: FactGroup,
        thread: tuple[tuple[str, str | None], ...] = (),
    ) -> Candidate:
        """Assembles one written question, whatever kind it was."""
        return Candidate(
            question_text=question.strip(),
            target_answer=answer.strip() or None,
            answerable=True,
            group=group,
            thread=thread,
            spec=plan.spec,
            planned_difficulty=plan.band,
        )

    @staticmethod
    def _conversation(thread: tuple[tuple[str, str | None], ...]) -> str:
        """Renders the turns already asked, oldest first."""
        turns = "\n".join(
            f"  Q: {question}\n  A: {answer or 'no answer in the material'}"
            for question, answer in thread
        )
        return f"CONVERSATION so far:\n{turns}"

    @staticmethod
    def _used(sample: FactGroup, numbered: list[int]) -> FactGroup:
        """The facts the model said its question needs, in citation order.

        A number the sample does not have is dropped rather than refused, and
        naming none is read as the first: the answer still has to rest on
        something, and a candidate with no facts is one the orphan trigger
        would never reach.
        """
        chosen = tuple(
            sample.facts[one - 1]
            for one in dict.fromkeys(numbered)
            if 1 <= one <= len(sample.facts)
        )
        return FactGroup(chosen or sample.facts[:1])

    @staticmethod
    def _prompt(sample: FactGroup) -> str:
        """Renders one sample: the facts numbered, then their passages.

        The passages are fenced and labelled, as extraction fences a heading
        trail, because a model shown them unlabelled asks about them instead
        of about the facts.
        """
        numbered = "\n".join(
            f"[{position}] {fact.statement}"
            for position, fact in enumerate(sample.facts, 1)
        )
        context = "\n\n".join(
            f"[{position}] {heading}{text}"
            for position, (heading, text) in enumerate(sample.context, 1)
        )
        return (
            f"FACTS - your question must be answered by these:\n{numbered}\n\n"
            f"PASSAGE(S) - context only, so you know what the material is "
            f"about. Phrase the question from these; never quote or name a "
            f"heading or a title out of them, and never ask about anything "
            f"here that the facts above do not state:\n{context}"
        )
