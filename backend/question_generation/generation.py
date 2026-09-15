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

The correction to that overshot, and the overshoot is worth recording. Told to
name what it was asking about, the writer named the document: `Laut den
'Risiken im Fokus 2026', ...`, `Gemäß der MaRisk, ...` - seven of ten in one
topic. Which is worse than vague, because a question carrying its own source
has already done the work it was meant to test. Nobody asks a service desk a
question while telling it which file to open. Naming the SUBJECT and naming
the SOURCE are different things, and only the first is wanted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.client import Client
from question_generation.models import Candidate, FactGroup

#: Recorded in the log beside every question written with the prompts below.
#: Bumped whenever one changes what a question is: two prompts are two
#: datasets, as with extraction.
PROMPT_VERSION = "4"

_ASK = """You write one test question for measuring a document-search chatbot.

You are given numbered FACTS drawn from a corpus, and the PASSAGE each came
from. The question must be answered by the facts. The passage is there so you
know what the material is about - use it to phrase the question, never as
something to ask about and never as something to cite.

Write the question somebody who needs this information would actually type.
They have not read the passage. They do not know which document answers them -
finding that out is the whole reason they are asking.

Rules, all of them mandatory:

- NEVER SAY WHERE THE ANSWER IS. No "According to the annual report", no "Laut
  dem Rundschreiben", no "Gemäß der MaRisk", no "in this circular", no naming
  or quoting a document, a report, a section or a heading. Nobody asks a
  service desk a question while telling it which file to open. A question that
  cites its own source has already done the work it was meant to test, and it
  is thrown away.

- NAME THE SUBJECT, NOT THE SOURCE. These are different things, and the
  difference is the whole rule. The subject is what the question is about -
  the thing being regulated, the duty, the party, the period. The source is
  which document says it. Name the first, never the second.

    NAME:   the supervised institution, the model type, the fee, the year the
            figure is from, the kind of risk, the authority whose duty it is
    NEVER:  the report it appears in, the circular that sets it, the section
            heading above it, the title of the material

- A question that opens with a bare "What" or "Which" and names nothing at
  all is still no good. Say what you are asking about - just not where to
  look it up.

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
         (names nothing at all, and no short answer is right)
  WRONG  question: "How quickly must support respond?"
         answer:   "must respond within 48 hours"
         (the answer is an action, not a thing)
  WRONG  question: "According to the service agreement, how long is allowed
                    for answering a standard support request?"
         (names its SOURCE. The person asking does not know there is a
          service agreement - that is what they are trying to find out)
  WRONG  question: "Under 'Support > Response times', what is the limit for a
                    standard request?"
         (the heading. Same failure, wearing a different hat)

  RIGHT  question: "How long is allowed for answering a standard support
                    request?"
         answer:   "48 hours"
         facts:    [1]

  RIGHT  question: "How long is allowed for answering a standard support
                    request, and how long for an urgent one?"
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
- NAME THE SUBJECT, NEVER THE SOURCE. Say what you are asking about - the
  party, the duty, the period, the thing being regulated - so the question
  reads as though it belongs to this material, because a question about an
  unrelated subject tests nothing: any chatbot declines that one. But NEVER
  say which document, report, circular or section it would be in. Nobody asks
  a service desk a question while telling it which file to open.
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
- ASK FOR ONE CHECKABLE VALUE, and answer with a SHORT NOUN PHRASE. Never a
  sentence, never anything with a verb in it.
- NEVER SAY WHERE THE ANSWER IS. No naming or quoting a document, a report, a
  section or a heading, here as anywhere else.
- DO NOT REPEAT a question already in the conversation, and do not ask one
  the last answer already gave. It has to want something new.
- Write in the language of the facts.
- `facts` is the NUMBERS of the facts your question needs.

Worked example.

  [1] A standard support request is answered within 48 hours.
  [2] An urgent support request is answered within 4 hours.

  CONVERSATION:
    Q: How long does the service agreement allow for answering a standard
       support request?
    A: 48 hours

  WRONG  "How long does the service agreement allow for a standard request?"
         (already asked)
  WRONG  "What is a support request?"
         (not in the facts)

  RIGHT  question: "And for an urgent one?"
         answer:   "4 hours"
         facts:    [2]
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

    def follow_up(
        self, sample: FactGroup, thread: tuple[tuple[str, str | None], ...]
    ) -> Candidate:
        """Asks the model for the question somebody would ask next.

        Written from the same sample as the thread it joins, so a follow-up
        is about the same material rather than a fresh question that happens
        to come after one.

        Only answerable questions get follow-ups. A thread whose first turn
        has no answer has nothing to follow on from - the chatbot was
        supposed to say it did not know - and asking a second question after
        that measures nothing.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        written = self._client.answer(
            system=_FOLLOW,
            user=f"{self._prompt(sample)}\n\n{self._conversation(thread)}",
            shape=_Answered,
        )
        return Candidate(
            question_text=written.question.strip(),
            target_answer=written.answer.strip() or None,
            answerable=True,
            group=self._used(sample, written.facts),
            thread=thread,
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
