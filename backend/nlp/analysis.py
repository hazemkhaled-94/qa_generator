"""Reading claims, sentences and vocabulary out of text with spaCy."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from functools import lru_cache
from importlib.metadata import version

from spacy.tokens import Doc, Span

from nlp.language import URL
from nlp.models import Claim, Sentence
from nlp.pipelines import pipeline
from settings import optional

#: Parts of speech kept as vocabulary. Everything else is grammar, which is
#: what a stopword list used to be needed for.
_CONTENT = frozenset({"NOUN", "PROPN", "ADJ"})

#: Parts of speech a noun is tagged with, in the languages that capitalise one.
_NOMINAL = frozenset({"NOUN", "PROPN"})


#: How many texts to hand spaCy at once.
_BATCH = 64

#: A web address. Markdown renders a link as [text](url), and the brackets
#: stop the tokenizer recognising the address as one token, so it splits on
#: the punctuation inside and offers every path segment as a word. Shared
#: with the detector, which must read the same text this does.
_URL = URL

_WHITESPACE = re.compile(r"\s+")

#: Sentence punctuation the tokenizer keeps inside a token's surface form. The
#: German tokenizer reads a year ending a sentence as one token, `2026.`, and
#: the same year anywhere else as `2026`.
_TRAILING = ".,;:"


@lru_cache(maxsize=1)
def _capitalises_nouns() -> frozenset[str]:
    """Reads the languages whose every noun is written with a capital.

    Optional: a deployment configured for languages that capitalise nothing
    names none, and no token is judged on its case.
    """
    named = optional("NLP_CAPITALISED_NOUNS") or ""
    return frozenset(code.strip().lower() for code in named.split(",") if code.strip())


def _bare(form: str) -> str:
    """Trims the sentence punctuation a token carried into its surface form.

    Trailing only, so `12,5` and `12.5` stay the different values they are.
    """
    return form.rstrip(_TRAILING) or form


def normalised(text: str) -> str:
    """Collapses spacing and case, for comparing two strings as words.

    `casefold` rather than `lower`, because this corpus is partly German and
    only casefold folds ß to ss - "STRASSE" and "Straße" are the same word.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


#: Recorded on every fact, beside the model and the prompt: the parser
#: decides the verdicts, so two parsers are two datasets.
VERSION = version("spacy")


#: Dependency labels that hang a clause off another one rather than standing
#: it beside. A finite verb under one of these is a modifier, not a second
#: assertion: German `rc` (relative), `oc` (clausal object), `mo` (adverbial),
#: `sb` (clausal subject), `re` (reported), and the UD names for the same.
#: Coordination is deliberately absent - `cj` and `conj` join two claims.
_SUBORDINATE = frozenset(
    {
        "rc",
        "oc",
        "mo",
        "sb",
        "re",
        "relcl",
        "ccomp",
        "xcomp",
        "advcl",
        "csubj",
        "acl",
    }
)


def _dependent(token) -> bool:
    """Whether a token sits inside a clause hung off another one."""
    return token.dep_ in _SUBORDINATE or any(
        parent.dep_ in _SUBORDINATE for parent in token.ancestors
    )


def _predicates(span: Doc | Span) -> int:
    """Counts the claims a span makes, as its independent finite verbs.

    A finite verb inside a relative or other subordinate clause is a
    modifier of something already said rather than a second thing said:
    "TPI NEXT gehört zu den Prozessmodellen, die die Testprozessverbesserung
    unterstützen" carries two finite verbs and one claim. Coordination is
    the other case and does count: two main clauses joined by "und" are two
    claims.

    A span carrying a finite verb makes at least one claim. A parse that
    hangs every clause off something else has mis-read the sentence, which
    is not the same as a sentence that asserts nothing.
    """
    finite = [token for token in span if "Fin" in token.morph.get("VerbForm", [])]
    return len([token for token in finite if not _dependent(token)]) or int(
        bool(finite)
    )


def _verbs(span: Doc | Span) -> int:
    """Counts verbs of any form, which is how an action is told from a thing."""
    return sum(1 for token in span if token.pos_ in ("VERB", "AUX"))


#: Tag prefixes of an interrogative word. Penn tags English ones WDT, WP, WP$
#: and WRB; STTS tags German ones PWS, PWAT and PWAV. Read off the tagger
#: rather than from a word list per language, so a pipeline added to
#: NLP_MODELS brings its own.
_INTERROGATIVE = ("W", "PW")


def _interrogative(token) -> bool:
    """Whether a token is a question word, under either naming.

    Two readings because no one of them covers every pipeline. Penn and STTS
    say so in the tag, and the English model leaves the morphology empty;
    a Universal Dependencies tagset says so in `PronType=Int` and its tags
    carry no such prefix. Either answer is taken, so a language whose
    pipeline uses only one of the two conventions still reports its question
    words.
    """
    return token.tag_.startswith(_INTERROGATIVE) or "Int" in token.morph.get(
        "PronType", []
    )


def interrogatives(text: str, language: str | None) -> tuple[str, ...]:
    """The question words one question uses.

    More than one is two questions joined by a conjunction - `Welche
    Vorschriften gelten für X und wie hoch ist die Gebühr für Y?` - which a
    chatbot can answer half of, and which nobody types.

    A finite-verb count does not separate those: `Welche Arten von Kryptowerten
    gelten als reguliert?` carries two and is one question. Measured over 17
    real questions, the question words separated 16.
    """
    return tuple(
        token.text for token in pipeline(language)(text) if _interrogative(token)
    )


def _units(span: Doc | Span) -> tuple[str, ...]:
    """Collects the values a statement may not introduce on its own.

    Numbers and proper nouns only. Both are things a writer reports rather
    than chooses, so one appearing in a statement and nowhere in the sentence
    it cites was invented.

    Three things are deliberately left out. Common nouns, because a statement
    is supposed to name its subject in its own words. Verbs, because the
    statement is written rather than quoted, so its predicate is the model's
    to choose - "Die Digitalisierung schreitet fort" drawn from
    "fortschreitender Digitalisierung" is the same claim and shares no verb
    lemma. And named entities, because the tagger's entity decision moves with
    the surrounding words, so the two sides never agree: "Umbrüchen" came back
    a MISC entity in a short statement and no entity at all in the longer
    sentence it was drawn from.

    A number keeps its surface form, because 12,5 and 12.5 are different
    values. A proper noun is lemmatised, so an inflection is not an addition.
    """
    found = set()
    for token in span:
        if token.like_num:
            found.add(_bare(token.text).casefold())
        elif token.pos_ == "PROPN":
            found.add(token.lemma_.casefold())
    return tuple(sorted(found))


#: Dependency labels for a pronoun that stands in for no antecedent: German
#: `ph` (Platzhalter) and `ep` (expletive es), and English `expl`. "Es gibt
#: Best Practices", "Es besteht Potenzial" and "There is potential" name
#: nothing and leave nothing dangling.
_EXPLETIVE = frozenset({"ph", "ep", "expl"})


def _refers(token) -> bool:
    """Reports whether a pronoun points at something outside the statement.

    A reflexive belongs to its own verb - `sich` in `ergibt sich aus` - and an
    expletive stands in for no antecedent, so neither leaves a reader without
    a subject.
    """
    if token.pos_ != "PRON" or token.dep_ in _EXPLETIVE:
        return False
    if "Yes" in token.morph.get("Reflex"):
        return False
    return "3" in token.morph.get("Person") or "Dem" in token.morph.get("PronType")


def _references(span: Doc | Span) -> tuple[str, ...]:
    """Collects the pronouns that leave a statement dependent on its context."""
    return tuple(sorted({t.text.casefold() for t in span if _refers(t)}))


def demonstratives(text: str, language: str | None) -> tuple[str, ...]:
    """The words in a text that point at something outside it.

    `PronType=Dem` off the morphology, which every Universal Dependencies
    tagset marks and so no word list per language is needed. Determiners as
    well as pronouns: `diesen beiden Lehrplänen` and `these two syllabi`
    point outward through the determiner, and the noun beside it is what
    makes the pointing look harmless.

    A measurement and not a verdict. Pointing is only a fault when there is
    nothing in the text to point AT, and half the questions carrying one set
    a case up first and then refer back to it - `Wenn ein KI-System ...,
    wie wird dieses Problem eingeordnet?` is self-contained. Deciding which
    is which is a reading, so this only says a pointer is present and the
    verifier says whether it lands.
    """
    return tuple(
        sorted(
            {
                token.text.casefold()
                for token in pipeline(language)(text)
                if "Dem" in token.morph.get("PronType", [])
            }
        )
    )


def _foreign(token, language: str) -> bool:
    """Whether a token is a foreign word read as a noun of this language.

    A language named in NLP_CAPITALISED_NOUNS writes every noun with a
    capital, so a lower-case one is a word from another language: the German
    pipeline tags the English `the`, `and` and `of` as proper nouns, which
    puts them in the German vocabulary as subjects.

    A lower-case German adjective the tagger reads as a noun goes with them.
    Measured over 200 German passages: 7,807 lemmas fell to 7,782, of which
    ten were German.
    """
    return (
        language in _capitalises_nouns()
        and token.pos_ in _NOMINAL
        and token.text[:1].islower()
    )


def _lemmas(document: Doc) -> list[str]:
    """Collects the content lemmas of a document.

    A web address is not vocabulary. Its path segments arrive tagged as nouns
    and proper nouns, and one repeated link put `publikationen`, `fokus`, `im`
    and `en` among the most widespread terms in the corpus.
    """
    addresses = [match.span() for match in _URL.finditer(document.text)]
    return [
        token.lemma_.casefold()
        for token in document
        if token.pos_ in _CONTENT
        and not _foreign(token, document.lang_)
        # The lemma, not the token: the lemma is what is stored, and a
        # lemmatiser handed a word from another language returns the
        # placeholder "--" for a token that is itself perfectly alphabetic.
        and token.lemma_.isalpha()
        and len(token.lemma_) > 1
        and not any(start <= token.idx < end for start, end in addresses)
    ]


def _sentences(document: Doc) -> list[Sentence]:
    """Reads the sentences out of a parsed document."""
    return [
        Sentence(
            index=index,
            start=sentence.start_char,
            end=sentence.end_char,
            text=sentence.text,
            predicates=_predicates(sentence),
        )
        for index, sentence in enumerate(document.sents)
    ]


def sentences(text: str, language: str | None) -> list[Sentence]:
    """Splits a passage into sentences, each located in the passage text."""
    return _sentences(pipeline(language)(text))


def vocabulary(text: str, language: str | None) -> frozenset[str]:
    """Every form a word of this text appears in, lemma and surface alike.

    What a statement's units are checked for presence against. Presence is
    asked rather than like-for-like, because the two sides are parsed
    separately and the tagger does not label a word the same way in a short
    statement as in the longer sentence it came from.
    """
    document = pipeline(language)(text)
    return frozenset(
        variant.casefold()
        for token in document
        for form in (token.text, token.lemma_)
        for variant in (form, _bare(form))
        if variant
    )


def content(text: str, language: str | None) -> frozenset[str]:
    """The content lemmas of a text: what it is about, and nothing else.

    The nouns, proper nouns and adjectives, lemmatised, which is the same
    reading the topic model is fitted over. Verbs and grammar are left out,
    so two inflections of one answer give the same set - which is what makes
    it possible to ask whether two phrasings say the same thing without
    asking whether they are spelled the same.
    """
    return frozenset(_lemmas(pipeline(language)(text)))


def claim(text: str, language: str | None) -> Claim:
    """Reads what one written statement asserts."""
    document = pipeline(language)(text)
    return Claim(
        predicates=_predicates(document),
        units=_units(document),
        references=_references(document),
        verbs=_verbs(document),
    )


def read(
    texts: Iterable[str], language: str | None
) -> Iterator[tuple[list[Sentence], list[str]]]:
    """Reads sentences and vocabulary for many passages, in order."""
    for document in pipeline(language).pipe(texts, batch_size=_BATCH):
        yield _sentences(document), _lemmas(document)
