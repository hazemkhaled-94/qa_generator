"""The things the nlp package hands back.

Plain values, and no spaCy import: a caller that only needs the shape of a
sentence should not load a pipeline to get it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    """One sentence of a passage, located in the passage text."""

    index: int
    start: int
    end: int
    text: str
    predicates: int


@dataclass(frozen=True)
class Claim:
    """What a statement asserts, and what it leaves unsaid."""

    predicates: int
    units: tuple[str, ...]
    references: tuple[str, ...]
    #: Verbs of any form, finite or not. `predicates` counts the finite ones,
    #: which is how many claims a statement makes; this counts all of them,
    #: which is how a phrase that names a thing is told from one that
    #: describes an action. An infinitive makes no claim and is still not an
    #: answer: `Verwarnungen aussprechen` has no predicate and one verb.
    verbs: int = 0


#: What each configured language is called, for a prompt that has to name it.
#:
#: A prompt telling a model to write "in the language of the excerpt" asks it
#: to infer the answer from the material, and on a corpus that is 1,444
#: German passages against 51 English it infers German: every validated fact
#: drawn from an English passage in one run came back in German, and nothing
#: in extraction checks a language, so the drift reached question generation
#: as a German question labelled English and was thrown out as `malformed`.
#:
#: Named rather than left as a code because that is what the instruction
#: says: "Write in English" is an instruction, "write in en" is a riddle.
_NAMED = {"de": "German", "en": "English"}


def named(language: str | None) -> str:
    """What to call one language in a prompt.

    Falls back to the code itself, which is the honest thing for a language
    `NLP_MODELS` has gained and this has not: an ISO 639-1 code is still an
    instruction a model can follow, where a wrong name is not. A language of
    None falls back to nothing, and the caller leaves the sentence out.
    """
    if not language:
        return ""
    return _NAMED.get(language.lower(), language)
