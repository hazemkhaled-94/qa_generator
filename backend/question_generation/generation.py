"""Writing one question from a sample of facts, answerable or deliberately not.

Both kinds come from here because both are one call against one sample with
one shape back, and splitting them would duplicate the rendering, the client
and the provenance to change the system prompt.

The writer is shown the passage each fact came from, and says which facts its
question needs. Neither was true of the first version of this module, and both
failures had the same cause: a single atomic statement is one triple, so the
only question that can be built out of it is that statement with one part
replaced by a question word. `Geopolitische Konflikte schüren Unsicherheit`
became `Was schüren geopolitische Konflikte?`, over and over. The passage is
what a question can be phrased from; the fact is what it has to be answered
by.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.client import Client
from question_generation.models import Candidate, FactGroup

#: Recorded in the log beside every question written with the prompts below.
#: Bumped whenever one changes what a question is: two prompts are two
#: datasets, as with extraction.
PROMPT_VERSION = "3"

_ASK = """You write one test question for measuring a document-search chatbot.

You are given numbered FACTS drawn from a corpus, and the PASSAGE each came
from. The question must be answered by the facts. The passage is there so you
know what the material is about and what it calls things - use it to phrase
the question, never as something to ask about.

Write the question somebody who needs this information would actually type.
They have not read the passage. They are searching a corpus for an answer.

Rules, all of them mandatory:

- NAME WHAT YOU ARE ASKING ABOUT, as a searcher would have to. Say which
  institution, which document, which rule, which year. A question that opens
  with a bare "What" or "Which" and no such anchor is not a question anybody
  could type into a corpus of thousands of pages.

- DO NOT TURN THE FACT INTO A QUESTION. Taking the sentence and replacing one
  part with a question word is the failure this whole task is about. If your
  question is the fact's own words in the fact's own order, throw it away and
  ask what a person would ask instead.

- ASK FOR ONE CHECKABLE VALUE. Something a person could mark right or wrong at
  a glance: how many, how much, by when, who, which one, what limit. A
  question asking what something "must provide", "covers" or "includes" has no
  answer anybody can score, however well it names its subject.

- THE ANSWER IS A SHORT NOUN PHRASE, and a few words at most: a value, an
  amount, a date, a name, a limit, a share. Never a sentence, never a clause,
  never anything with a verb in it. If the answer you want to write is a
  sentence, you asked too broad a question - ask for one of the things in that
  sentence instead.

- NEVER PUT THE ANSWER IN THE QUESTION, or the word the answer is a kind of.
  Asking "For which models do the requirements apply?" when the answer is
  "automated models" tests nothing.

- ONE question, ending in a question mark. One thing asked.

- Write in the language of the facts.

- `facts` is the NUMBERS of the facts your question needs. Use several only
  when the question genuinely cannot be answered without all of them - two
  facts about different subjects are two questions, not one. Most questions
  need one fact, and saying so is correct.

Worked example. Facts, under the heading "Support > Response times":

  [1] A standard support request is answered within 48 hours.
  [2] An urgent support request is answered within 4 hours.

  PASSAGE: Standard requests are answered within 48 hours on working days.
  Urgent requests are answered within 4 hours and may be raised by phone.
  These times are set out in the service agreement.

  WRONG  question: "Within how many hours is a standard request answered?"
         (the fact with its number deleted; nobody types this)
  WRONG  question: "What are the response times?"
         (names nothing, and no short answer is right)
  WRONG  question: "How quickly must support respond?"
         answer:   "must respond within 48 hours"
         (the answer is an action, not a thing)
  WRONG  question: "According to the service agreement, what answer times does
                    it set for support requests?"
         answer:   "Standard requests are answered within 48 hours on working
                    days and urgent requests within 4 hours."
         (names its subject, and still useless: the answer is a sentence, so
          nobody can mark a chatbot right or wrong against it)

  RIGHT  question: "How long does the service agreement allow for answering a
                    standard support request?"
         answer:   "48 hours"
         facts:    [1]

  RIGHT  question: "Under the service agreement, what are the answer times
                    for standard and for urgent support requests?"
         answer:   "48 hours and 4 hours"
         facts:    [1, 2]
"""

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
- NAME WHAT YOU ARE ASKING ABOUT, as a searcher would: the institution, the
  document, the rule, the year. It must read as though it belongs to this
  material, because a question about an unrelated subject tests nothing - any
  chatbot declines that one.
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

  RIGHT  "What answer time does the service agreement set for a standard
          support request raised on a public holiday?"
         (same subject, same shape, a condition the material does not cover)
"""


class _Answered(BaseModel):
    """A question the facts answer, as the model is asked to return it."""

    question: str = Field(
        description="ONE question, ending in a question mark, naming what it "
        "asks about the way a searcher would have to."
    )
    answer: str = Field(
        description="The answer as a short noun phrase, a few words at most: "
        "a value, an amount, a date, a name, a limit or a share. Never a "
        "sentence and never anything with a verb in it."
    )
    facts: list[int] = Field(
        default_factory=list,
        description="The NUMBERS of the facts this question needs. Usually "
        "one. Several only when it cannot be answered without all of them.",
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

    def write(self, sample: FactGroup, *, answerable: bool) -> Candidate:
        """Asks the model for one question, with or without an answer.

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
        if answerable:
            written = self._client.answer(
                system=_ASK, user=self._prompt(sample), shape=_Answered
            )
            return Candidate(
                question_text=written.question.strip(),
                target_answer=written.answer.strip() or None,
                answerable=True,
                group=self._used(sample, written.facts),
            )

        # Perturbed from one fact, not from the sample: moving a claim just
        # out of reach is a change to one claim, and a reader given two would
        # have two ways to notice.
        first = FactGroup(sample.facts[:1])
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
            f"PASSAGE(S) - what the material says and what it calls things. "
            f"Phrase the question from these; do not ask about anything here "
            f"that the facts above do not state:\n{context}"
        )
