"""The second model, asked to get the answer back out of the passages.

Three of the gates come out of one call here, and recoverability may cost a
second: recalling an answer cold and checking one put in front of you are
different tasks, and conflating them was rejecting four good questions for
every real disagreement. `read` asks for the answer; when that comes back
empty or different, `supports` shows the model the answer and asks only
whether the passages back it.

Two things are load-bearing. The verifier is a different model from the
writer, because a model marking its own work agrees with itself. And it sees
only the cited passages, never the corpus: what is being measured is the
dataset, not a retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from llm.client import Client

_VERIFY = """You check one test question for a document-search chatbot.

You are given some PASSAGES and a QUESTION. Do two separate things.

FIRST, answer the question using ONLY those passages. You have no other
knowledge.

- `in_passage` is true when the passages give the answer. READING IS ALLOWED:
  the answer may be spread over two sentences, worded differently from the
  question, or put together from two statements that are both in the
  passages. Doing that is reading them, not guessing.
- It is false when you would need something the passages do not contain - a
  fact you know from elsewhere, a number you would have to estimate or
  round, a party or a period they never name.
- `answer` is the answer, taken from the passages, when in_passage is true.
  Leave it empty otherwise. IN THE WORDS OF THE PASSAGES, and so in their
  language: an answer translated into English is compared against a German
  target and agrees with nothing.

Saying it is not in the passages is a correct answer whenever it is true, and
inventing one is the failure. But so is refusing an answer that IS there
because it took two sentences to find: look before you decline.

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


_SUPPORT = """You check whether some PASSAGES support one specific ANSWER to
one QUESTION.

You are not writing an answer and you are not judging whether it is the best
one. One thing only: do these passages say what this answer says?

- `supported` is true when they do. The wording does not have to match. A
  paraphrase, another inflection, a shorter form, or the same thing said
  across two sentences all count, and so does an answer that puts together
  two statements the passages both make.
- `supported` is false when the passages say something DIFFERENT - another
  number, another party, another period, another condition - or when they do
  not address it at all. A different number is never a paraphrase: if the
  answer says 4 hours and the passages say 48 hours, that is false.
- You have no knowledge outside the passages. An answer that is true in the
  world but that these passages do not give is not supported.

  TRUE:  passages "Anträge werden binnen 48 Stunden beantwortet."
         answer   "48 Stunden"
  TRUE:  passages "Der Standortleiter genehmigt jede Änderung."
         answer   "der Standortleiter"
  FALSE: passages "Anträge werden binnen 48 Stunden beantwortet."
         answer   "4 Stunden"
  FALSE: passages "Anträge werden binnen 48 Stunden beantwortet."
         answer   "binnen 48 Stunden, an Werktagen"
         (the working-day part is not in the passages)
"""


_COMPUTES = """You check whether one ANSWER can be WORKED OUT from some
PASSAGES.

The answer is deliberately not written in the passages. It is a total, a
count or a sum that somebody has to compute from figures that ARE there.
That it does not appear is expected and is not a reason to say no.

- `supported` is true when every figure the computation needs is in the
  passages AND the arithmetic gives this answer. Do the arithmetic.
- It is false when a figure is missing from the passages, or when they are
  all there and come to something else.
- You have no knowledge outside the passages. Do not supply a missing figure
  from anywhere else.

  TRUE:  passages "Der Nordstandort beschäftigt 40 Personen." and "Der
                   Südstandort beschäftigt 25 Personen."
         answer   "65"
  FALSE: same passages, answer "68"           (the arithmetic gives 65)
  FALSE: passages naming only the northern site, answer "65"
                                               (the other figure is missing)
"""


_FOLLOWS = """You check whether one ANSWER FOLLOWS from some PASSAGES.

The answer is deliberately not written in the passages. The passages state
the premises; the answer is what they come to when put together, or what a
rule in them says about a case they do not mention. That it does not appear
is expected and is not a reason to say no.

- `supported` is true when every premise the conclusion needs is in the
  passages AND the conclusion really follows from them. Reason it through.
- It is false when a premise is missing, when the passages settle the
  question differently, or when the answer needs something you know from
  outside them. A conclusion that is merely plausible does not follow.
- It is also false when the answer is simply STATED in the passages. Then
  nothing was derived and the question is a lookup wearing this kind's
  name.

  TRUE:  passages "Anträge werden binnen 48 Stunden beantwortet." and "Die
                   48 Stunden zählen nur Werktage."
         answer   "ein Freitagsantrag ist am Dienstag fällig, weil das
                   Wochenende nicht zählt"
  FALSE: same passages, answer "am Sonntag"     (the weekend does not count)
  FALSE: passages naming only the 48 hours, answer "am Dienstag"
                                                (the working-day premise is
                                                 not there)
"""


class _Supported(BaseModel):
    """Whether the passages back one proposed answer."""

    supported: bool = Field(
        description="True only when the passages say what the answer says. "
        "A different number, party or period is not support."
    )


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

    def computes(
        self,
        question: str,
        answer: str,
        passages: Sequence[str],
        thread: Sequence[tuple[str, str | None]] = (),
    ) -> bool:
        """Whether a derived answer follows from the figures in the passages.

        What an `aggregation` is judged by instead of recoverability. That
        type's answer is a total the material does not write down anywhere,
        so asking whether the passages state it refuses every one of them.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        return self._judge(_COMPUTES, question, answer, passages, thread)

    def follows(
        self,
        question: str,
        answer: str,
        passages: Sequence[str],
        thread: Sequence[tuple[str, str | None]] = (),
    ) -> bool:
        """Whether a conclusion follows from the premises in the passages.

        The entailment half of what `computes` does for arithmetic. An
        `implication` puts two statements together and an `application`
        puts a rule to a case, and in both the answer is absent from the
        material by construction - so recoverability, which asks whether
        the passages state it, would refuse every one of them.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        return self._judge(_FOLLOWS, question, answer, passages, thread)

    def supports(
        self,
        question: str,
        answer: str,
        passages: Sequence[str],
        thread: Sequence[tuple[str, str | None]] = (),
    ) -> bool:
        """Whether the passages back this exact answer, asked as a yes or no.

        The second half of the recoverability gate, and the reason that gate
        is two calls. Recalling an answer out of a passage cold is a harder
        task than checking one that is put in front of you, and the two were
        being conflated: over one corpus the verifier failed to recall an
        answer 76 times against 19 real disagreements about what the answer
        was, so four in five of the biggest rejection class were the task
        rather than the question.

        Only ever asked after `read` has failed, and only ever able to
        rescue: a question this refuses was already being refused.

        Raises:
            ModelUnavailable: If the model could not be reached or would not
                answer in the shape.
        """
        return self._judge(_SUPPORT, question, answer, passages, thread)

    def _judge(
        self,
        system: str,
        question: str,
        answer: str,
        passages: Sequence[str],
        thread: Sequence[tuple[str, str | None]],
    ) -> bool:
        """Puts one answer to the model as a yes or no under these rules.

        Written once because two gates ask it: whether the passages support
        an answer, and whether they let one be worked out. The two differ
        only in what they expect of an answer the passages do not contain.
        """
        numbered = "\n\n".join(
            f"[{position}] {passage}" for position, passage in enumerate(passages, 1)
        )
        conversation = (
            "\n".join(f"Q: {q}\nA: {a or 'not in the material'}" for q, a in thread)
            + "\n"
            if thread
            else ""
        )
        got = self._client.answer(
            system=system,
            user=(
                f"Passages:\n{numbered}\n\n{conversation}"
                f"Question: {question}\nAnswer: {answer}"
            ),
            shape=_Supported,
        )
        return got.supported

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
