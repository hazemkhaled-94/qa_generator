"""The gates that need neither a model nor an index.

Every one of these is a measurement: it reads the question, its answer and
the material it was drawn from, and returns a code to store with a reason to
log. Nothing here calls a model, which is what makes them the gates a
re-check can apply to a stored question - and why this module imports no
client.

Every gate that reads an answer reads it against the FORM the question's type
asked for. One rule for all of them was what made the set what it was: a value
may carry no verb, so applied to every answer that rule refused every why,
how, procedure and consequence question ever written.

There is no gate here for "it is not the kind of question it was asked to be",
and that is a finding rather than an omission. The verifier was asked it, as
`matches_intent`, and answered `true` for a bare `Wie viele Anlassprüfungen
wurden 2025 durchgeführt?` written into a `reason` slot - the same failure the
phrasing judgement had when it was a yes or no. What survives is structural:
a kind declares an answer form, and `wrong_form` reads the answer against it,
so a reason answered with a value is still refused. `question_type` records
what was ASKED for, as `planned_difficulty` does.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from database.qa_generator import AnswerForm, QuestionRejection
from nlp.analysis import claim, content, interrogatives, normalised, vocabulary
from nlp.language import detect
from question_generation.models import Neighbour

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
#:
#: Public because the reason the checker logs beside the verdict names it.
NAMES_SOMETHING = 3


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
        len(content(question, language)) >= NAMES_SOMETHING
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
