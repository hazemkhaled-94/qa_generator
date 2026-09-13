"""Reading claims, sentences and vocabulary out of text with spaCy."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import spacy
from spacy.tokens import Doc, Span

from nlp.models import Claim, Sentence
from nlp.pipelines import pipeline

#: Parts of speech kept as vocabulary. Everything else is grammar, which is
#: what a stopword list used to be needed for.
_CONTENT = frozenset({"NOUN", "PROPN", "ADJ"})

#: How many texts to hand spaCy at once.
_BATCH = 64

#: A web address. Markdown renders a link as [text](url), and the brackets
#: stop the tokenizer recognising the address as one token, so it splits on
#: the punctuation inside and offers every path segment as a word.
_URL = re.compile(r"https?://\S+|www\.\S+")

_WHITESPACE = re.compile(r"\s+")


def normalised(text: str) -> str:
    """Collapses spacing and case, for comparing two strings as words.

    `casefold` rather than `lower`, because this corpus is partly German and
    only casefold folds ß to ss - "STRASSE" and "Straße" are the same word.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


#: Recorded on every fact, beside the model and the prompt: the parser
#: decides the verdicts, so two parsers are two datasets.
VERSION = spacy.__version__


def _predicates(span: Doc | Span) -> int:
    """Counts finite verbs, which is how many claims a span makes."""
    return sum(1 for token in span if "Fin" in token.morph.get("VerbForm"))


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
            found.add(token.text.casefold())
        elif token.pos_ == "PROPN":
            found.add(token.lemma_.casefold())
    return tuple(sorted(found))


#: Dependency labels for a pronoun that stands in for no antecedent: German
#: `ph` (Platzhalter) and English `expl`. "Es besteht Potenzial" and "There is
#: potential" name nothing and leave nothing dangling.
_EXPLETIVE = frozenset({"ph", "expl"})


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
        form.casefold()
        for token in document
        for form in (token.text, token.lemma_)
        if form
    )


def claim(text: str, language: str | None) -> Claim:
    """Reads what one written statement asserts."""
    document = pipeline(language)(text)
    return Claim(
        predicates=_predicates(document),
        units=_units(document),
        references=_references(document),
    )


def read(
    texts: Iterable[str], language: str | None
) -> Iterator[tuple[list[Sentence], list[str]]]:
    """Reads sentences and vocabulary for many passages, in order."""
    for document in pipeline(language).pipe(texts, batch_size=_BATCH):
        yield _sentences(document), _lemmas(document)
