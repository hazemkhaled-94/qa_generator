"""Writing one question from a group of facts, answerable or deliberately not.

Both kinds come from here because both are one call against one group with
one shape back, and splitting them would duplicate the rendering, the client
and the provenance to change the system prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.client import Client
from question_generation.models import Candidate, FactGroup

#: Recorded in the log beside every question written with the prompts below.
#: Bumped whenever one changes what a question is: two prompts are two
#: datasets, as with extraction.
PROMPT_VERSION = "1"

_ASK = """You write one test question from the facts you are given.

The question measures whether a chatbot can find these facts in the corpus it
was given, so it has to be a question those facts - and only those facts -
answer.

Rules, all of them mandatory:
- ONE question, ending in a question mark.
- Answerable from the facts below and from nothing else. Never ask for
  something they do not state.
- The answer is SHORT: the value, the name, the date, the duty. Not a
  sentence restating the fact.
- NAME THE SUBJECT. A reader who cannot see the facts must be able to tell
  what is being asked about. Never "it", "this", or "the company" where the
  fact says which company.
- Do not hand the answer back in the question. A question answerable by yes
  tests nothing.
- Given SEVERAL facts, write the one question that needs ALL of them. If no
  such question exists, write one for the first fact and ignore the rest.
- Write in the language of the facts.

Worked example, from two facts:

  [1] A standard support request is answered within 48 hours.
  [2] An urgent support request is answered within 4 hours.

  WRONG  question: "What are the response times?"
         (needs neither fact in particular, and no short answer is right)
  WRONG  question: "Is a standard support request answered within 48 hours?"
         (answerable by yes, without reading anything)

  RIGHT  question: "Within how many hours is a standard support request
                    answered, and within how many an urgent one?"
         answer:   "48 hours and 4 hours"
"""

_PERTURB = """You write one test question that the material does NOT answer.

These measure whether a chatbot says it does not know instead of inventing an
answer. So the question has to be a plausible thing to ask of this material
and have no answer anywhere in it.

Take the fact below and move it just out of reach. Make ONE change:
- ask about a neighbouring thing the fact does not cover,
- ask for a detail of the same subject the fact does not state, or
- ask about a different party, place, period or category.

Rules, all of them mandatory:
- ONE question, ending in a question mark.
- It must READ as though it belongs to this document. A question about an
  unrelated subject tests nothing, because any chatbot declines that one.
- It must NOT be answerable from the fact. If reading the fact answers your
  question, you have written the wrong question.
- Do not ask something absurd and do not invent a thing that does not exist.
  Both are too easy to decline.
- Keep the subject named, as the fact names it.
- Write in the language of the fact.

Worked example:

  Fact: "A standard support request is answered within 48 hours."

  WRONG  "How long does a standard support request take to answer?"
         (that is the fact itself, and it is answered)
  WRONG  "What is the top speed of a swallow?"
         (nothing to do with this material; declining it tests nothing)

  RIGHT  "Within how many hours is a standard support request answered on a
          public holiday?"
         (same subject, same shape, a condition the material does not cover)
"""


class _Answered(BaseModel):
    """A question the facts answer, as the model is asked to return it."""

    question: str = Field(
        description="ONE question, ending in a question mark, answerable from "
        "the facts and naming what it asks about."
    )
    answer: str = Field(
        description="The short answer: the value, the name, the date or the "
        "duty. Not a sentence restating the fact."
    )


class _Unanswered(BaseModel):
    """A question the facts do not answer."""

    question: str = Field(
        description="ONE question, ending in a question mark, that reads as if "
        "it belongs to this material and that the fact does not answer."
    )


class QuestionWriter:
    """Writes one question per fact group with a served model.

    The model never sees the corpus, only the facts: a fact is already one
    claim standing on its own, which is what makes it something to write a
    question from rather than a passage to summarise.
    """

    def __init__(self, client: Client) -> None:
        """Initialises the writer with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model writing the questions."""
        return self._client.model

    def write(self, group: FactGroup, *, answerable: bool) -> Candidate:
        """Asks the model for one question, with or without an answer.

        Returns whatever came back, unjudged. An empty question is a
        candidate like any other: the gates reject it and the rejection is
        what the drop rate is measured from.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        if answerable:
            written = self._client.answer(
                system=_ASK, user=self._prompt(group), shape=_Answered
            )
            return Candidate(
                question_text=written.question.strip(),
                target_answer=written.answer.strip() or None,
                answerable=True,
                group=group,
            )

        # Perturbed from one fact, not from the group: moving a claim just
        # out of reach is a change to one claim, and a reader given two would
        # have two ways to notice.
        first = FactGroup(group.facts[:1])
        written = self._client.answer(
            system=_PERTURB, user=self._prompt(first), shape=_Unanswered
        )
        return Candidate(
            question_text=written.question.strip(),
            # An unanswerable question is scored on behaviour and not on
            # content, which the questions table enforces with a CHECK.
            target_answer=None,
            answerable=False,
            group=first,
        )

    @staticmethod
    def _prompt(group: FactGroup) -> str:
        """Renders one group's facts, numbered from one."""
        numbered = "\n".join(
            f"[{position}] {fact.statement}"
            for position, fact in enumerate(group.facts, 1)
        )
        return f"Facts:\n{numbered}"
