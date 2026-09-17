"""The gates a written question has to pass.

Applied cheapest first, because each one that fires saves the cost of those
behind it:

  malformed       structural, free
  answer_too_short  free: nothing worth scoring against
  answer_too_long   free: not the form of answer that was asked for
  wrong_form        free: a value carrying a verb, an explanation with none
  leaks_source    free where the title is quoted, the verifier's otherwise
  off_topic       free: an unanswerable question about nothing the material
                  mentions, which any chatbot declines
  duplicate       one index probe against the questions already accepted
  answerable      the same probe, when an unanswerable question has a twin
                  the corpus does answer
  compound        it asks two things, so half an answer is neither right
                  nor wrong
  unanchored      a question nobody could have asked without the passage
  recoverable     the answer is not in the evidence the question cites
  answerable_elsewhere  a lemma probe and a second call, on the unanswerable
                  share only: a passage it does not cite answers it

Three of those come out of one model call. Recoverability may cost a second:
recalling an answer cold and checking one put in front of you are different
tasks, and conflating them was rejecting four good questions for every real
disagreement. `read` asks for the answer; when that comes back empty or
different, `supports` shows the model the answer and asks only whether the
passages back it. The units check guards that pass, because an opinion must
not be able to confirm a number the material never gave.

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
import re
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


def _misfits(
    answer: str,
    form: str,
    bounds: Mapping[str, tuple[int, int]],
    verbs: bool,
) -> tuple[str, str] | None:
    """Why this answer is not of this form, or None when it is.

    Two readings: how long it is, and what it is made of. A value carrying a
    verb is a fragment of the sentence rather than the thing asked for -
    `Verwarnungen aussprechen`, `nachvollziehbar zu begruenden`. Any verb and
    not only a finite one, because those two are infinitives and make no
    claim: a phrase can describe an action without asserting it. Only for an
    answer of more than one word, because a single word is a thing by
    construction and `de_core_news_md` tags `dreihundert` as a verb.

    An explanation carrying no verb is the opposite failure: a why or a how
    answered with a noun phrase has not been answered.
    """
    low, high = bounds.get(form, BOUNDS[AnswerForm.VALUE])
    if len(answer) < low:
        return (
            QuestionRejection.ANSWER_TOO_SHORT,
            (
                f"the target answer {answer!r} is {len(answer)} characters and a "
                f"{form} answer is held to at least {low}"
            ),
        )
    if len(answer) > high:
        return (
            QuestionRejection.ANSWER_TOO_LONG,
            (
                f"the target answer {answer!r} is {len(answer)} characters and a "
                f"{form} answer is held to at most {high}"
            ),
        )
    if form == AnswerForm.VALUE and len(answer.split()) > 1 and verbs:
        return (
            QuestionRejection.WRONG_FORM,
            (
                f"the target answer {answer!r} describes an action rather "
                f"than naming a thing, and this question asked for a value"
            ),
        )
    if form == AnswerForm.EXPLANATION and not verbs:
        return (
            QuestionRejection.WRONG_FORM,
            (
                f"the target answer {answer!r} names a thing rather than "
                f"explaining anything, and this question asked for an explanation"
            ),
        )
    return None


#: What separates the items of a list, for the shape hint below. Punctuation
#: only, and deliberately no conjunction: `und`, `and`, `et` and the rest are
#: a per-language list, and nothing else in this stage holds one. A two-item
#: list joined by a bare conjunction is therefore read as a value, which
#: costs it the looser comparison and not the question - the entailment pass
#: is what catches those.
_ITEMS = re.compile(r"[,;]")

#: How many items punctuation has to separate before an answer is taken for a
#: list. Two commas, so `A, B and C` is a list and `Berlin, 2025` is not.
_LISTED = 3


def _looks_like(answer: str, verbs: bool) -> str:
    """The form an answer appears to take, read before any type is consulted.

    Which matters because the forms are tried in order and the first that
    fits wins: a three-item list is inside a value's 80 characters and
    carries no verb, so it passed as a value and was then held to a value's
    comparison, which wants every one of its lemmas back. The hint puts the
    form the answer actually has first, where its type allows that form.
    """
    if len(_ITEMS.split(answer)) >= _LISTED:
        return AnswerForm.LIST
    if verbs and len(answer.split()) > 1:
        return AnswerForm.EXPLANATION
    return AnswerForm.VALUE


def fitting(
    answer: str,
    forms: Sequence[str],
    bounds: Mapping[str, tuple[int, int]],
    language: str | None,
) -> tuple[str, tuple[str, str] | None]:
    """The first of these forms the answer fits, and what stopped it if none.

    A type asks for one form and will take the others it declares. What
    shape an answer has is decided by the material, not by the question:
    `Welche Werkzeuge werden empfohlen?` is a factoid whose answer is three
    tools, and a `condition` is answered `immer` when that is the condition.
    Holding each to the single form its type asked for refused both, and
    then held them to a `value`'s stricter comparison on the way out.

    The complaint reported is the asked-for form's, not the last one tried:
    that is the form the writer was told to produce, so it is the one a
    person reading the rejection needs to see.
    """
    verbs = bool(claim(answer, language).verbs)
    looks = _looks_like(answer, verbs)
    tried = (
        (looks, *(one for one in forms if one != looks)) if looks in forms else forms
    )
    first: tuple[str, str] | None = None
    for form in tried:
        failed = _misfits(answer, form, bounds, verbs)
        if failed is None:
            return form, None
        if form == forms[0]:
            first = failed
    return forms[0], first or _misfits(answer, forms[0], bounds, verbs)


def structural(
    *,
    question_text: str,
    target_answer: str | None,
    answerable: bool,
    language: str | None,
    statements: Sequence[str] = (),
    forms: Sequence[str] = (AnswerForm.VALUE,),
    bounds: Mapping[str, tuple[int, int]] | None = None,
    titles: Sequence[str] = (),
) -> tuple[tuple[str, str] | None, str]:
    """The gates that need neither a model nor an index, and the form it took.

    Returns the code to store with the reason to log, or None when it
    passes, and the answer form the question turned out to have. The several
    ways of being malformed share one code, because none of them measures
    anything a report would group on and the reason belongs where a person
    debugging reads it. The ones that do get a code of their own - an answer
    outside the bounds of its form, an answer of the wrong form, and a
    question quoting the title of its own source - are the ones a report
    should count.

    `forms` is every shape the question's type will take its answer in, the
    one it asked for first. The form reported back is the one the answer
    actually fits, and it is what the later gates and the stored row read:
    a factoid answered with a list is compared as a list.
    """
    asked = question_text.strip()
    form = forms[0] if forms else AnswerForm.VALUE
    if not asked:
        return (QuestionRejection.MALFORMED, "the model returned no question"), form
    if not asked.endswith("?"):
        return (
            QuestionRejection.MALFORMED,
            "it does not end in a question mark",
        ), form
    if asked.count("?") > 1:
        return (
            QuestionRejection.MALFORMED,
            (
                f"it asks {asked.count('?')} things; a chatbot answering one of "
                f"them is neither right nor wrong"
            ),
        ), form
    if answerable and not (target_answer or "").strip():
        return (
            QuestionRejection.MALFORMED,
            (
                "it is answerable and carries no target answer, so nothing could "
                "be scored against it"
            ),
        ), form
    if not answerable and (target_answer or "").strip():
        return (
            QuestionRejection.MALFORMED,
            (
                "it is unanswerable and carries a target answer; an unanswerable "
                "question is scored on behaviour, not on content"
            ),
        ), form
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
        ), form

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
        ), form

    # Whichever of the type's forms the answer fits, with its bounds and its
    # rule about verbs. QUESTIONS_ANSWER_CHARS sets both ends of each form,
    # so an answer is never scored against something too thin to be worth
    # scoring - `7`, `8%`, `Nein` - nor against an essay where a value was
    # asked for.
    answer = (target_answer or "").strip()
    if answer:
        form, failed = fitting(answer, forms, bounds or BOUNDS, language)
        if failed:
            return failed, form

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
        ), form

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
        ), form
    return None, form


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


#: How much of what an unanswerable question is about has to occur in the
#: material it was drawn from. QUESTIONS_OFF_TOPIC_OVERLAP is where it is set.
OFF_TOPIC_OVERLAP = 0.3


def on_topic(
    question: str,
    language: str | None,
    lemmas: frozenset[str],
    floor: float = OFF_TOPIC_OVERLAP,
) -> bool:
    """Whether an unanswerable question is about the material at all.

    The gate the unanswerable questions did not have. Everything else here
    reads a question against an answer, and an unanswerable one has none, so
    the only thing that could refuse one was the verifier finding an answer
    after all - which passed 87% of them against 30% of the rest. They were
    not better; they were less tested.

    What makes one worthless is being easy to decline. `What is the top speed
    of a swallow?` asked of a testing syllabus is declined by any chatbot and
    measures nothing; the question worth asking is the one that reads as
    though this corpus should answer it and does not. The prompt says so and
    nothing checked it.

    Measured on content lemmas against the lemmas chunking stored for the
    passages the question was drawn from - the same vocabulary the topics
    were fitted over, so no text is parsed twice and the two sides are the
    same reading. A question naming nothing the material names shares none of
    them; a perturbation of one of its facts shares most of its subject and
    differs in the one thing that was moved out of reach.

    A question with no content lemmas at all is left to `unanchored`, which
    is the gate for that. And a passage carrying no lemmas abstains rather
    than refusing everything drawn from it: `passages.lemmas` is nullable,
    a passage chunked before the column existed has none, and a measurement
    with nothing to measure against is not evidence of anything.
    """
    asked = content(question, language)
    if floor <= 0 or not asked or not lemmas:
        return True
    return len(asked & lemmas) / len(asked) >= floor


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


def asserted(target: str, passages: Sequence[str], language: str | None) -> bool:
    """Whether every number, date and name the answer asserts is in the passages.

    The guard on the entailment pass. A model asked "do these passages
    support this answer" says yes more readily than it recalls one, and the
    one error that must not get through is a number the material never gave:
    `4 hours` confirmed against a passage that says 48 puts a question in the
    benchmark that marks a correct chatbot wrong.

    Extraction's `unsupported_addition` check, pointed at an answer instead
    of a statement, and read against the passages rather than against what
    came back - which is the whole point here, because nothing came back.
    """
    units = claim(target, language).units
    if not units:
        return True
    found = {_folded(one) for one in vocabulary("\n".join(passages), language)}
    return all(_folded(unit) in found for unit in units)


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
        elsewhere=None,
        elsewhere_passages: int = 0,
        off_topic_overlap: float = OFF_TOPIC_OVERLAP,
        entail: bool = True,
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

        `elsewhere` is the corpus-wide passage probe the unanswerable gate
        reads, taking the question's lemmas, its language, the passages it
        already cites and a limit. A callable and not the catalogue, for the
        same reason `nearest` is one. None, or a limit of 0, leaves an
        unanswerable question judged against its own passages alone.
        """
        self._embedder = embedder
        self._verifier = verifier
        self._nearest = nearest
        self._judge_phrasing = judge_phrasing
        self._bounds = dict(bounds or BOUNDS)
        self._overlap = overlap
        self._long_answer_chars = long_answer_chars
        self._threshold = threshold
        self._elsewhere = elsewhere
        self._elsewhere_passages = elsewhere_passages
        self._off_topic_overlap = off_topic_overlap
        self._entail = entail

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
        failed, form = structural(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            language=candidate.group.language,
            statements=candidate.group.statements,
            forms=candidate.spec.forms,
            bounds=self._bounds,
            titles=candidate.group.titles,
        )
        if failed:
            return self._verdict(candidate, failed, None, self._long_answer_chars, form)

        # Free, and only an unanswerable question can fail it: an answerable
        # one is about its material by construction, because the answer came
        # out of it.
        if not candidate.answerable and not on_topic(
            candidate.question_text,
            candidate.group.language,
            candidate.group.lemmas,
            self._off_topic_overlap,
        ):
            return self._verdict(
                candidate,
                (
                    QuestionRejection.OFF_TOPIC,
                    (
                        "it names almost nothing the passages it was drawn from "
                        "name, so any chatbot declines it and declining it "
                        "proves nothing"
                    ),
                ),
                None,
                self._long_answer_chars,
                form,
            )

        embedding = self._embedder.embed(candidate.question_text)
        failed = near_verdict(
            self._near(embedding, seen),
            answerable=candidate.answerable,
            threshold=self._threshold,
        )
        if failed:
            return self._verdict(
                candidate, failed, embedding, self._long_answer_chars, form
            )

        return self._verdict(
            candidate,
            self._round_trip(candidate, form),
            embedding,
            self._long_answer_chars,
            form,
        )

    def _round_trip(self, candidate: Candidate, form: str) -> tuple[str, str] | None:
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
            # A derived answer is not in the passages and is not meant to
            # be. Asking recall for it is asking the verifier to contradict
            # the type's own directive - `aggregation` says the total must
            # be something the material does not write down - and it did:
            # 7 of its 12 questions went as not_recoverable, and the type
            # accepted at 8% where the rest of them averaged 40%.
            if candidate.spec.derived:
                return self._computable(candidate, target)
            if read.recovered is None:
                if self._backed(candidate, target):
                    return None
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    "the verifier could not find the answer in the cited passages",
                )
            if not agrees(
                read.recovered,
                target,
                candidate.group.language,
                form,
                self._overlap,
                candidate.question_text,
            ):
                if self._backed(candidate, target):
                    return None
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
        return self._corpus(candidate)

    def _computable(self, candidate: Candidate, target: str) -> tuple[str, str] | None:
        """Whether a derived answer follows from the figures in the passages.

        The gate an `aggregation` gets instead of recoverability, because
        the two ask opposite questions. Recoverability asks whether the
        passages STATE the answer; this type is defined by their not doing
        so, and the answer is right when the figures it needs are there and
        the arithmetic comes out.

        Still a model's judgement, and still the verifier's rather than the
        writer's, so nothing here marks its own work.
        """
        if not target:
            return (
                QuestionRejection.NOT_RECOVERABLE,
                "it carries no total to check the passages against",
            )
        if self._verifier.computes(
            candidate.question_text,
            target,
            candidate.group.passages,
            candidate.thread,
        ):
            return None
        return (
            QuestionRejection.NOT_RECOVERABLE,
            (
                f"the total {target!r} does not follow from the figures in the "
                f"cited passages"
            ),
        )

    def _backed(self, candidate: Candidate, target: str) -> bool:
        """Whether the entailment pass rescues an answer recall did not find.

        Two guards, both of which have to hold. Every number, date and name
        the target asserts must occur in the passages, which no opinion can
        overrule: that is what keeps `4 hours` from being confirmed against
        a passage that says 48. Then the verifier is shown the answer and
        asked the narrower question.

        It can only ever accept. A question reaching here was already being
        rejected, so a pass that says nothing leaves the verdict where it
        was and costs one call on the failures alone.
        """
        if not self._entail or not target:
            return False
        passages = candidate.group.passages
        if not asserted(target, passages, candidate.group.language):
            return False
        backed = self._verifier.supports(
            candidate.question_text, target, passages, candidate.thread
        )
        if backed:
            log.info(
                "keeping %r: recall missed it but the passages support %r",
                candidate.question_text,
                target,
            )
        return backed

    def _corpus(self, candidate: Candidate) -> tuple[str, str] | None:
        """Asks whether the rest of the corpus answers what its passages do not.

        The one claim in this dataset that is about the whole corpus rather
        than about the passages a question cites. Every other gate is right
        to read the citation alone - what is being measured is the dataset
        and not a retriever - but "there is no answer to this anywhere" is
        not a property of two passages, and a question wrongly carrying it
        marks a correct chatbot wrong, which is the one failure the gates
        exist to stop.

        Costs one probe and one call, on the unanswerable share only.
        """
        if self._elsewhere is None or self._elsewhere_passages <= 0:
            return None
        language = candidate.group.language
        passages = self._elsewhere(
            sorted(content(candidate.question_text, language)),
            language,
            [passage.id for passage in candidate.group.resting],
            self._elsewhere_passages,
        )
        if not passages:
            return None
        read = self._verifier.read(candidate.question_text, passages, candidate.thread)
        if read.recovered is None:
            return None
        return (
            QuestionRejection.ANSWERABLE_ELSEWHERE,
            (
                f"a passage it does not cite answers it, with "
                f"{read.recovered!r}, so the corpus is not silent about this"
            ),
        )

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
        form: str | None = None,
    ) -> CheckedQuestion:
        """Assembles one judged question, kept whichever way it went.

        `form` is the one the answer turned out to fit, of those its type
        will take. Stored rather than the asked-for one, because it is what
        the gates read it against and a row saying otherwise could not be
        re-checked to the same verdict.
        """
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
            answer_form=(form or candidate.spec.form) if candidate.answerable else None,
            planned_difficulty=candidate.planned_difficulty,
        )
