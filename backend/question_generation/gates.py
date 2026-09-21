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

The gate for "it is not the kind of question it was asked to be" is
structural, and only ever structural. The verifier was once asked the
judgement, as `matches_intent`, and answered `true` for a bare `Wie viele
Anlassprüfungen wurden 2025 durchgeführt?` written into a `reason` slot - it
fired zero times over 71 questions while the kind was plainly wrong on six of
27, which is the same failure the phrasing judgement had when it was a yes or
no. So a model is not asked again. What a rule can settle is settled: a kind
declares an answer form and `wrong_form` reads the answer against it, an
`entity` question has to ask after a party, and an `enumeration` has to answer
with more than one thing. Every other kind records what was ASKED for, as
`planned_difficulty` does.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import combinations

from database.qa_generator import AnswerForm, QuestionRejection
from nlp.analysis import (
    claim,
    content,
    interrogatives,
    normalised,
    phrases,
    vocabulary,
)
from nlp.language import detect
from nlp.pipelines import pipeline
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


def subject(question: str, language: str | None) -> str:
    """What a question names, as the spans it names it in.

    The reading that used to be a field on the verifier's answer, where it
    was asked for as "copy the thing the question is about, word for word".
    Copying a span out of a question is a parse, and asking a model to do it
    inside a call about the passages cost it three of nineteen labelled
    cases - twice by naming something for a question that names nothing, and
    once by finding nothing in a question that does.

    Every content-bearing noun phrase rather than the best one: a question is
    allowed to be about more than one thing, `Wie unterscheiden sich in
    Abschnitt 2.2 die Themen zur Testschätzung und zur Fehlerbehebung?` is
    about two, and choosing between them is a judgement nothing here needs.

    The numbers and proper nouns go on the end, because a chunker does not
    always keep one: `die Norm ISO/IEC 20246` chunks as `die Norm ISO IEC`
    and the identifier is the whole of what that question names.

    Empty when the question names nothing, and `anchored` is what decides
    that rather than anything about the chunks: `What specific components are
    included?` and `Für welche Kriterien gelten die Anforderungen?` both
    chunk to something and neither names a thing.

    Nothing gates on this. `anchored` is the verdict; this is what a person
    reading the rejection needs in order to disagree with it.
    """
    if not anchored(question, language):
        return ""
    named = list(phrases(question, language))
    written = normalised(" ".join(named))
    named.extend(
        unit for unit in claim(question, language).units if unit not in written
    )
    return " ".join(named)


#: Words that name a part of a document rather than a thing in the world, and
#: so say WHERE an answer is once one is numbered. `Abschnitt 2.2` is a
#: citation; `Abschnitt` alone is not, which is why every one of these is
#: read with an identifier beside it.
_DIVISIONS = frozenset(
    {
        "abschnitt",
        "kapitel",
        "anhang",
        "artikel",
        "ziffer",
        "absatz",
        "paragraf",
        "paragraph",
        "section",
        "chapter",
        "appendix",
        "annex",
        "clause",
        "article",
    }
)

#: Words naming a document itself. Read by lemma and never as a substring,
#: because German compounds: `Risikobericht` is a thing the corpus is about
#: and `Bericht` is a thing the corpus IS, and a substring test cannot tell
#: them apart.
_DOCUMENTS = frozenset(
    {
        "lehrplan",
        "norm",
        "bericht",
        "dokument",
        "handbuch",
        "richtlinie",
        "verordnung",
        "rundschreiben",
        "leitfaden",
        "vereinbarung",
        "syllabus",
        "report",
        "document",
        "circular",
        "regulation",
        "standard",
        "guideline",
        "manual",
        "agreement",
        "specification",
    }
)

#: Lemmas that attribute what follows them to a source. `laut dem
#: Jahresbericht` says where the answer is whatever the noun turns out to be,
#: so this one needs nothing beside it.
_ATTRIBUTIONS = frozenset({"laut", "gemäß", "ausweislich", "according", "per"})

#: A bracketed reference as a bibliography writes one: `[R22]`, `[12]`.
_BRACKETED = re.compile(r"\[[A-Za-z]{0,3}\s?\d{1,4}[a-z]?\]")

#: A year, which is what turns a proper noun into a citation when it sits
#: directly after one: `Beck 2003`.
_CITED_YEAR = re.compile(r"^(1[89]|20)\d{2}$")


def _identifier(token) -> bool:
    """Whether a token numbers the division or document before it."""
    return token.like_num or bool(any(char.isdigit() for char in token.text))


#: How far past an attribution to look for the thing being attributed to.
#: One short noun phrase, which is what `laut dem Jahresbericht 2025` is.
_ATTRIBUTED_WITHIN = 4


def _names_something(following: Sequence) -> bool:
    """Whether what an attribution points at is named rather than pointed at."""
    for token in following[:_ATTRIBUTED_WITHIN]:
        if token.pos_ == "PROPN" or _identifier(token):
            return True
        if token.lemma_.casefold() in _DOCUMENTS:
            return True
    return False


def cites_source(question: str, language: str | None) -> bool | None:
    """Whether a question says WHERE its answer is, read off its parse.

    True when a pattern settles it, and **None when nothing here does** -
    never False. Absence of a pattern is not evidence a question names no
    source: `Welche Reviewverfahren beschreibt die Norm ISO/IEC 20246?` names
    one and `Warum wird ISO/IEC/IEEE 29119-4 erwähnt?` does not, and the two
    differ by which of the standard and the answer the question is about.
    That is a reading, so it is left to whatever holds an opinion.

    What the patterns settle is the half a model was measurably bad at. Over
    the nineteen labelled cases gpt-4.1 missed four, and three of them are
    `Kapitel 5`, `[R22]` and `Beck 2003` - a numbered division, a bracketed
    reference and an author with a year. None of those needs a reading.
    """
    document = pipeline(language)(question)
    if _BRACKETED.search(question):
        return True

    tokens = list(document)
    for position, token in enumerate(tokens):
        lemma = token.lemma_.casefold()
        following = tokens[position + 1] if position + 1 < len(tokens) else None

        # `laut`, `gemäß`, `according to` - but only where what is attributed
        # to is NAMED. `laut diesen Angaben` attributes to something the
        # question points at and never names, which is a question failing to
        # stand on its own rather than one citing a source, and the two are
        # different gates.
        if lemma in _ATTRIBUTIONS and _names_something(tokens[position + 1 :]):
            return True
        # `Abschnitt 2.2`, `Kapitel 5`, `section 4`.
        if lemma in _DIVISIONS and following is not None and _identifier(following):
            return True
        # `Beck 2003`: a proper noun with a bare year against it. Adjacency
        # is load-bearing - `reported to the site manager in 2025` names a
        # party and a period and no source, and the preposition is what
        # says so.
        if (
            token.pos_ == "PROPN"
            and following is not None
            and _CITED_YEAR.match(following.text.strip())
        ):
            return True
        if lemma in _DOCUMENTS:
            # `die Norm ISO/IEC 20246`: the document named and numbered.
            if following is not None and _identifier(following):
                return True
            # `in diesem Lehrplan`: the question pointing at the corpus it
            # is asked of. Only for a word that names a document - `in
            # diesem Zusammenhang` points at the discussion, not at a file.
            if any(
                "Dem" in child.morph.get("PronType", []) for child in token.children
            ):
                return True
    return None


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


def asks_the_same(first: str, second: str, language: str | None) -> bool:
    """Whether two questions ask the same KIND of thing about their subject.

    Read off the question words, as `compound` reads them. Two questions
    sharing none are not the same question however close their vectors are:
    `Wofür ist der Lehrplan gedacht?` and `Welche Personengruppen dürfen
    den Lehrplan verwenden?` are about one subject and want different
    answers.

    That is not a hypothetical. Over the 454 duplicates one corpus refused,
    **318 - 70% - share no question word with the twin they were refused
    against**: `warum` against `welche`, `wie` against `warum`. A sentence
    embedding encodes what a question is ABOUT, and two questions about one
    subject sit on top of each other whatever they ask for.

    Abstains where either question uses no question word at all, which
    leaves those to the threshold as before: a measurement with nothing to
    measure is not evidence that they differ.
    """
    first_words = {word.casefold() for word in interrogatives(first, language)}
    second_words = {word.casefold() for word in interrogatives(second, language)}
    if not first_words or not second_words:
        return True
    return bool(first_words & second_words)


def _unalike(threshold: float) -> float:
    """How close two questions asking DIFFERENT things have to be.

    Halfway from the threshold to identical, so it moves with the setting
    rather than being a second number to keep in step with it: at the 0.93
    QUESTIONS_DUPLICATE_COSINE ships with, a question asking something else
    has to reach 0.965 before it is a duplicate anyway.
    """
    return threshold + (1.0 - threshold) / 2


def near_verdict(
    near: Neighbour | None,
    *,
    answerable: bool,
    threshold: float,
    question: str = "",
    language: str | None = None,
) -> tuple[str, str] | None:
    """What the nearest accepted question says about this one.

    One probe answering two gates. A near twin of an accepted question is a
    duplicate and adds nothing to the benchmark; a near twin of an accepted
    *answerable* question, when this one is meant to be unanswerable, is a
    question the corpus demonstrably does answer.

    The probe is a measurement and `asks_the_same` is the veto on it, which
    is the shape the phrasing gates already use. Only the duplicate half is
    vetoed: `answerable_after_all` says the corpus answers something this
    close, and that holds whatever the two questions ask for.
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
    # Asking something else is not refused outright, it is held to a higher
    # bar. Two question words can want the same answer - `How heavy is the
    # device?` against `What is the weight?` - so a different one is
    # evidence and not proof. The false positives measured over one corpus
    # sat at 0.937 to 0.948, and a real paraphrase sits above this.
    if (
        question
        and near.similarity < _unalike(threshold)
        and not asks_the_same(question, near.question_text, language)
    ):
        return None
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


#: Shortest and longest explanation, as QUESTIONS_EXPLANATION_CHARS sets
#: them. The fallback for a re-check reading a row whose settings are not to
#: hand, as BOUNDS is.
EXPLANATION_CHARS = (150, 900)

#: How much of an explanation may be lemmas the target answer already had
#: before it has said nothing the target did not. An explanation is SUPPOSED
#: to restate its answer - it is an answer written out - so this is set where
#: only a near-copy fails it, not where a faithful expansion does.
_RESTATES = 0.9


def _asserts(text: str, language: str | None) -> tuple[str, ...]:
    """The units an explanation claims, minus the ones prose counts with.

    `asserted` reads every number and proper noun, which is right for a
    target answer: that is short, and every unit in one is the answer. Prose
    counts as it goes - `the slower of the two`, `both editions`, `the first
    of these` - and none of those is a claim the material has to carry. The
    explanation of a 48-hour reply time was refused for saying `two` about
    two figures it had just quoted correctly.

    So a spelled-out numeral is dropped and a figure is kept. `48` is a
    claim the passages can contradict; `two` is how a sentence is built.
    Read off `like_num`, which every Universal Dependencies tagset marks, so
    no list of number words per language is needed.

    Proper nouns stay. A name an explanation invents is the failure this is
    for, and no language writes one to hold a sentence together.
    """
    counted = {
        token.text.casefold()
        for token in pipeline(language)(text)
        if token.like_num and not any(char.isdigit() for char in token.text)
    }
    return tuple(
        unit for unit in claim(text, language).units if unit.casefold() not in counted
    )


def explains(
    explanation: str,
    target: str,
    passages: Sequence[str],
    language: str | None,
    span: tuple[int, int] = EXPLANATION_CHARS,
) -> bool:
    """Whether the long answer is worth showing a reader.

    Three readings, and none of them is recoverability. That gate compares
    the TARGET against what a verifier independently wrote back, and its
    denominator is the target's own content lemmas - so every lemma an
    answer gains is another one the verifier has to reproduce, and an answer
    written to teach somebody is an answer it refuses. Measured before this
    column existed: explanations sat at a median of 157 characters against a
    ceiling of 600, because the prompt asked for one or two sentences and
    every worked example answered in a fragment.

    So the explanation is held to what it is actually for:

    - LONG ENOUGH to be a reading rather than the target said twice, and
      short enough to stay an answer rather than becoming the passage.

    - ASSERTING NOTHING THE MATERIAL DOES NOT. Every figure, date and name
      has to be in the cited passages - `asserted`'s reading, over the units
      `_asserts` keeps. This is the only one of the three that can put a
      wrong claim in front of a reader, and it is why the gate is not simply
      a length check.

    - SAYING SOMETHING THE TARGET DID NOT. An explanation whose every
      content lemma is already in the target has expanded nothing. The floor
      is high on purpose: restating the answer is most of what an
      explanation does, and only a near-copy should fail here.

    Returns True where there is no explanation at all. A question written
    before this column, or by a writer that returned none, is not rejected
    for it - the column is nullable and the checker decides whether one was
    required.
    """
    if not explanation.strip():
        return True
    low, high = span
    if not low <= len(explanation) <= high:
        return False
    units = _asserts(explanation, language)
    if units:
        found = {_folded(one) for one in vocabulary("\n".join(passages), language)}
        if not all(_folded(unit) in found for unit in units):
            return False
    said = content(explanation, language)
    if not said or not target:
        return bool(said)
    return len(said & content(target, language)) / len(said) < _RESTATES


#: The question words that ask for a party rather than a thing. An `entity`
#: question asks who, and a language marks that in one word - the same
#: reading `compound` makes of interrogatives, which measured 16 of 17.
#:
#: A list per language and not a feature, because no Universal Dependencies
#: tagset marks animacy on an interrogative: German `wer` and English `who`
#: are both PRON/PronType=Int, exactly as `was` and `what` are. Short, and
#: the two languages NLP_MODELS configures are in it.
_AGENTS = frozenset({"wer", "wem", "wen", "wessen", "who", "whom", "whose"})


def asks_for_an_agent(question: str, answer: str, language: str | None) -> bool:
    """Whether an `entity` question asks after a party at all.

    The type is defined as "who does, decides, owns or must be told
    something", and over one measured run 208 of 244 accepted `entity`
    questions - 85% - answered with no party in sight: `Womit können
    Test-Chartas erstellt werden?` answered `mit Flipcharts und
    Tabellenkalkulationen` is a factoid wearing the label.

    That matters beyond the label, because `cognitive_level` is derived from
    the type and inherits whatever the type got wrong.

    Two ways to pass, because the question is not the only place the party
    shows. Either the question uses an agent interrogative, or the answer
    names a person or an organisation - which is spaCy's `PER` and `ORG`,
    and so needs no word list of roles. A role that is neither, `the site
    manager`, is caught by the first reading, because a question whose
    answer is a role asks `who`.

    This is the structural half of the gate that was deleted. The verifier
    was once asked whether a question was the kind it had been planned as
    and fired zero times over 71 questions while the kind was plainly wrong
    on six of 27; a rule reads what a rule can settle, and nothing here is
    a matter of taste.
    """
    if any(word.casefold() in _AGENTS for word in interrogatives(question, language)):
        return True
    return any(
        entity.label_ in ("PER", "PERSON", "ORG")
        for entity in pipeline(language)(answer).ents
    )


#: The fewest items an enumeration answers with. Its own directive says "if
#: there is only one item, you asked the wrong question for this kind", and
#: nothing enforced it.
_ENUMERATED = 2


def enumerates(answer: str, language: str | None) -> bool:
    """Whether an `enumeration` answer holds a set rather than one thing.

    Counted on noun phrases rather than on separators, because a list is
    written with commas in one language and with none in another, and
    `Systemtests, der Testmanager und der Technical Test Analyst` is three
    items however it is punctuated.
    """
    return len(phrases(answer, language)) >= _ENUMERATED


def moves_on(cited: Sequence[int], root: Sequence[int]) -> bool:
    """Whether a follow-up reaches a fact the question before it did not.

    A thread is supposed to walk through the material, and this is the
    reading of whether it did. Over one measured run 739 of 1,047 accepted
    follow-ups - 70.6% - cited nothing new, because `_followups` was handed
    the root's own sample and the type cycle then asked for the same fact in
    another shape. That is what produced threads like `Warum erstellt ein
    Team ein Teamvokabular?` / `Was können die Teammitglieder vermeiden?` /
    `Unter welchen Umständen können die Teammitglieder Missverständnisse
    vermeiden?` - one fact, three interrogatives, and nothing learned after
    the first.

    Read on facts and not on wording, because wording is what a type cycle
    changes and the defect is that the material underneath did not.
    """
    return bool(set(cited) - set(root))


def same_material(cited: Sequence[int], parent: Sequence[int]) -> bool:
    """Whether a follow-up rests on a passage the turn before it used.

    The other way a thread fails: not asking the same thing again, but
    changing the subject. `Warum sollen beim Mehrfachbedingungstest
    Testfälle entworfen werden?` followed by `Warum wurde TTA-2.6.1
    entfernt?` is not a conversation, and 147 accepted follow-ups shared no
    passage with their parent.

    Read on passages rather than on shared words. A follow-up MAY lean on
    the conversation - `Und bei einem dringenden?` is a good one and carries
    no content word at all - so a lexical reading would refuse exactly the
    elliptical follow-ups the thread exists to produce. What has to stay
    constant is the material, not the vocabulary.

    Abstains where either side rests on nothing, which a question with no
    citation does; the orphan trigger is what handles that.
    """
    if not cited or not parent:
        return True
    return bool(set(cited) & set(parent))


#: How much of what a question asks ABOUT has to occur in the passages
#: before the entailment pass may rescue an answer recall did not find.
#: QUESTIONS_ENTAILMENT_OVERLAP is where it is set.
ENTAILMENT_OVERLAP = 0.3


def about(
    question: str,
    passages: Sequence[str],
    language: str | None,
    floor: float = ENTAILMENT_OVERLAP,
) -> bool:
    """Whether the passages are about what the question asks about.

    The guard the entailment pass was missing, and the LMT case is why it
    exists. `Was ist ein Liquiditätsmanagementtool?` answered `eine
    einjährige Rückgabefrist` is confirmed by any judge asked "do these
    passages say what this answer says", because the passages do say there
    is a one-year redemption period. What they never mention is a
    Liquiditätsmanagementtool - so the answer is supported and answers a
    different question, which is exactly the defect the golden set keeps
    that case for.

    Asking about the ANSWER cannot catch it. `asserted` already checks every
    number and name the answer asserts, and this answer invents none: it is
    lifted out of the passage. What is absent is the thing the QUESTION is
    about, so that is what this reads.

    Measured on content lemmas, which is the same reading `on_topic` uses
    for the other direction. A question naming nothing is let through, for
    the same reason a passage with no lemmas abstains there: a measurement
    with nothing to measure is not evidence of anything.
    """
    asked = content(question, language)
    if floor <= 0 or not asked:
        return True
    found = content("\n".join(passages), language)
    if not found:
        return True
    return len(asked & found) / len(asked) >= floor


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


#: A figure as either convention writes one: 1.234,56 and 1,234.56 are the
#: same value, and 65 is itself.
_FIGURE = re.compile(
    r"(?<![\w.,])\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?(?![\w])|(?<![\w.,])\d+(?![\w.,])"
)

#: The most figures a sum is searched over, and the most terms in one. The
#: search is every combination of up to this many, so both are what keeps it
#: from being exponential. Four covers every aggregation this corpus asked
#: for; a total of five figures nobody wrote down is not a question anybody
#: would type.
_FIGURE_CAP = 40
_TERM_CAP = 4

#: The range a bare four-digit number is read as a year in. Two of them
#: summed is the arithmetic that produced the answer 4034, which is exactly
#: right and measures nothing.
_YEAR = range(1900, 2101)


def figures(text: str) -> list[float]:
    """Every number in a text, as a value.

    A group of exactly three digits after the last separator is a thousands
    group in both conventions, so 1.234 and 1,234 are 1234. Anything else
    after the last separator is a fraction, so 12,5 and 12.5 are 12.5. A
    number that resolves to neither is left out rather than guessed at.
    """
    found: list[float] = []
    for match in _FIGURE.finditer(text):
        raw = match.group()
        head, _, tail = raw.rpartition(",") if "," in raw[-4:] else raw.rpartition(".")
        try:
            if not head or len(tail) == 3:
                found.append(float(raw.replace(".", "").replace(",", "")))
            else:
                found.append(float(f"{head.replace('.', '').replace(',', '')}.{tail}"))
        except ValueError:
            continue
    return found


def adds_up(answer: str, passages: Sequence[str]) -> bool | None:
    """Whether the answer is a total of figures the passages state.

    The check an `aggregation` gets instead of asking a model to do the
    arithmetic. A total is arithmetic, and arithmetic is the one judgement
    here that has a right answer rather than a likely one: a model asked
    whether 40 and 25 come to 65 is being asked to agree, and it agrees with
    68 often enough to matter.

    Returns:
        True when some combination of at most four of the figures in the
        passages sums to the answer, False when none does, and None when the
        answer is not a single figure - a total in words, a range, or a date -
        which is not this check's to judge and falls back to the model.
    """
    wanted = figures(answer)
    if len(wanted) != 1:
        return None
    total = wanted[0]

    available = [one for text in passages for one in figures(text)][:_FIGURE_CAP]
    if not available:
        return False

    # Stated outright is not an aggregation, but it is not wrong arithmetic
    # either; `_derived` is the gate that refuses a lookup wearing this name.
    for size in range(1, _TERM_CAP + 1):
        for combination in combinations(available, size):
            if abs(sum(combination) - total) < 1e-9:
                return not _summed_years(combination, size)
    return False


def _summed_years(combination: tuple[float, ...], size: int) -> bool:
    """Whether a total is two or more years added together.

    Refused rather than accepted, because the arithmetic is right and the
    quantity is not one: a year is a point on a calendar, and two of them
    come to nothing. This corpus produced `Welche Jahreszahl ergibt sich aus
    dem Veroeffentlichungsjahr ... und der Versionsjahreszahl ...?` answered
    `4034`, which the model checking the arithmetic correctly confirmed.
    """
    return size > 1 and all(
        one.is_integer() and int(one) in _YEAR for one in combination
    )
