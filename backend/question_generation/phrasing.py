"""The judgements about a question alone, asked one at a time.

Three of these used to ride along in the verifier's reading call, and the
measurement that split them out is in `evaluation/README.md`: over nineteen
labelled cases gpt-4.1 answered them perfectly in English and at chance in
German - 7/7 against 6/12 - while answering the other half of the same call,
recovering an answer out of the passages, well in both. A judgement sharing a
call with a harder task is answered in the language the harder task is
thinking in.

Two of the three no longer need a model at all. `subject` is a span copied
out of the question, which `gates.subject` reads off the parse and gets right
on 19 of 19 where the model got 16. `names_its_source` is settled by
`gates.cites_source` wherever a pattern settles it, which is every case the
model got wrong.

What is left is the residue, and it is asked here. One question per call,
because the interference this module exists to undo is what happens when that
is not true.

Each call is also **gated by a measurement**, so the common question costs
nothing: a question carrying no pointing word is not asked whether its
pointers land, and one whose source-naming a rule already settled is not
asked about that either.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from llm.client import Client, ModelUnavailable

log = logging.getLogger(__name__)

#: Bumped whenever a prompt here changes what it asks for. Version 2 recast
#: every worked example into the neutral domain the rest of the prompts use:
#: the examples had been taken from the corpus being debugged against, which
#: is the one thing extraction's own prompt warns about.
PROMPT_VERSION = "3"

_SOURCE = """ROLE
You are a question reviewer. You judge ONE property of ONE question and
nothing else about it.

ACTION
Say whether the question says WHERE its answer is - naming or quoting a
document, a report, a circular, a regulation by name, a section or a heading.

STEPS
1. Read the question.
2. List every document, report, regulation, section or heading it names.
3. For each one, decide whether the question treats it as WHERE THE ANSWER
   IS, or as THE THING IT IS ASKING ABOUT.
4. Answer true if any one of them is named as the source, false otherwise.

CONTEXT
Naming a party, a duty, a period or a thing being governed is NOT naming a
source. Only saying which material holds the answer is. The hard cases are
the ones where a document is named.

The question may be in any language, and the examples below are in English
and German only because these instructions are. Judge what the question
does, not what it is written in.

EXAMPLES
TRUE:  "According to the service agreement, how long may a reply take?"
TRUE:  "Under 'Support > Response times', what is the urgent reply time?"
TRUE:  "What does section 4 say about weekend cover?"
TRUE:  "Welche Antwortzeiten schreibt die Norm XY-4711 vor?"
       (the norm is being cited as what states the answer)
FALSE: "How long is allowed for answering an urgent support request?"
FALSE: "Who approves a change to the shift plan?"
FALSE: "How many faults were reported to the site manager in 2025?"
       (the site manager is who they were reported TO, not a source)
FALSE: "Wer trägt die Verantwortung für den vierteljährlichen Risikobericht?"
       (the report is what the question is ABOUT, not where the answer is)
FALSE: "Warum wird die Norm XY-4711 in diesem Zusammenhang erwähnt?"
       (the standard is the subject being asked about, not the source.
        The pair above and this one have the same shape and differ only by
        which of the norm and the answer the question is about)

FORMAT
Return `names_its_source`, true or false, and nothing else."""

_CONTAINED = """ROLE
You are a question reviewer. You judge ONE property of ONE question and
nothing else about it.

ACTION
Say whether every POINTING WORD listed has something INSIDE THE QUESTION
ITSELF to point at.

STEPS
1. Read the question, then the pointing words found in it.
2. Take each pointing word in turn and look for the thing it points at.
3. Decide whether that thing was introduced by this question, or whether it
   sits outside the question.
4. Answer true only when every pointing word lands inside the question.

CONTEXT
The asker has not seen the material. A pointing word that reaches outside the
question reaches nothing they could know.

The test is not whether the pointing word is ordinary. It is whether the
thing it points at was introduced by this question. A question that sets a
case up and then refers back to it is self-contained, however much it reads
like a reference.

The question may be in any language. Judge what it does, not what it is
written in.

EXAMPLES
TRUE:  "If a system meets its target by editing the stored score instead of
        doing the task, how is this behaviour classified?"
       (pointing word: "this". The question set the behaviour out first, so
        it points at something present.)
TRUE:  "Wenn ein Standort eine zentrale Warteschlange einführt, ist dieser
        Ansatz für einen großen Standort geeignet?"
       (pointing word: "dieser". The approach was just named.)
FALSE: "What changed in this version?"
       (pointing word: "this". No version is named anywhere in the question.)
FALSE: "Wie groß ist der Unterschied zwischen den beiden Ausgaben?"
       (pointing word: "beiden". WHICH two? The question names neither, and
        only somebody holding the material could know. Naming them -
        "zwischen der Ausgabe 2022 und der von 2024" - is a question.)
FALSE: "Wie unterscheiden sich die Antwortzeit und ein Teammitglied laut
        diesen Angaben?"
       (pointing word: "diesen". No Angaben are named.)

FORMAT
Return `self_contained`, true or false, and nothing else."""


class _NamesItsSource(BaseModel):
    """Whether one question says where its answer is."""

    names_its_source: bool = Field(
        description="True only when the question says WHICH MATERIAL holds "
        "the answer. Naming a party, a duty, a period, or the thing the "
        "question is about is not naming a source."
    )


_NAMES = """ROLE
You are a question reviewer. You judge ONE property of ONE question and
nothing else about it.

ACTION
Say whether the question names something a person searching a large corpus
could have typed it about - a thing, a party, a duty, a period, a place.

STEPS
1. Read the question.
2. Take every noun in it in turn.
3. Decide whether each is a bare word - components, requirements, criteria,
   items - with nothing saying whose or which.
4. Answer false only when EVERY noun in the question is such a bare word.

CONTEXT
The asker has not read the material and does not know which document answers
them. A question that names nothing is one only somebody already holding the
passage could have asked.

The test is not whether the question is detailed. It is whether every noun in
it is a bare word with nothing saying whose or which.

The question may be in any language. Judge what it does, not what it is
written in.

EXAMPLES
TRUE:  "How long is allowed for answering a standard support request?"
TRUE:  "Why must a request be confirmed in writing?"
TRUE:  "What does the device weigh?"
       (thin, but it names a device and a property of it)
FALSE: "What specific components are included?"
       (components of what? nothing says)
FALSE: "Für welche Kriterien gelten die Anforderungen?"
       (which criteria, whose requirements? nothing says)
FALSE: "Which items are covered?"

FORMAT
Return `names_something`, true or false, and nothing else."""

#: The user messages, as the catalogue records them and `PhrasingJudge`
#: renders them.
_USER = "QUESTION: {{question}}"
_USER_CONTAINED = "QUESTION: {{question}}\n\nPOINTING WORDS found in it: {{pointers}}"


class _NamesSomething(BaseModel):
    """Whether one question names anything a searcher could have typed."""

    names_something: bool = Field(
        description="True when the question names a thing, party, duty, "
        "period or place. False only when every noun in it is a bare word "
        "with nothing saying whose or which."
    )


class _SelfContained(BaseModel):
    """Whether one question's pointing words land inside it."""

    self_contained: bool = Field(
        description="True when every pointing word listed has something "
        "inside the question itself to point at. False when one of them "
        "reaches outside the question."
    )


class PhrasingJudge:
    """Whatever is left of the phrasing judgements after the rules."""

    def __init__(self, client: Client) -> None:
        """Initialises the judge with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model answering, recorded beside what it answered."""
        return self._client.model

    def names_its_source(self, question: str) -> bool | None:
        """Whether a question says where its answer is.

        Only ever asked for the residue: `gates.cites_source` settles the
        patterns, and this is what is left when no pattern applies.

        Returns:
            What the model said, or None when it could not be reached. None
            is not False - a judgement nobody made must not reject a
            question - and the caller is what decides that.
        """
        return self._asked(
            _SOURCE,
            _USER.replace("{{question}}", question),
            _NamesItsSource,
            "names_its_source",
        )

    def names_something(self, question: str) -> bool | None:
        """Whether a question names anything a searcher could have typed.

        Only ever asked where the parse has already called the question
        thin, which is what keeps it off the common candidate. It is the
        second holder of a two-part verdict and not the verdict: the
        measurement alone rejects `Why must a request be confirmed in
        writing?`, which is a question somebody would type, and an opinion
        alone rejected `According to the ECB and NCAs, who conducts the due
        diligence check?`, which names two parties.

        Returns:
            What the model said, or None when it could not be reached.
        """
        return self._asked(
            _NAMES,
            _USER.replace("{{question}}", question),
            _NamesSomething,
            "names_something",
        )

    def self_contained(self, question: str, pointers: Sequence[str]) -> bool | None:
        """Whether a question's pointing words land inside it.

        The pointers are handed over rather than left to be found: the parse
        already located them, and a model asked to both find and judge them
        is a call doing two things. A question carrying none is never asked,
        because there is nothing to rule on.

        Returns:
            What the model said, or None when it could not be reached or
            there was nothing to judge.
        """
        if not pointers:
            return None
        asked = _USER_CONTAINED.replace("{{question}}", question).replace(
            "{{pointers}}", ", ".join(pointers)
        )
        return self._asked(_CONTAINED, asked, _SelfContained, "self_contained")

    def _asked(self, system: str, user: str, shape, field: str) -> bool | None:
        """One question, one judgement, and None when it could not be had."""
        try:
            answered = self._client.answer(
                system=system, user=user, shape=shape, prompt_version=PROMPT_VERSION
            )
        except ModelUnavailable as exc:
            # Logged and abstained rather than raised. These are opinions
            # about phrasing, and the run is about the answers: losing one
            # must not fail a question the passages support.
            log.warning("no %s judgement for %r: %s", field, user, exc)
            return None
        return bool(getattr(answered, field))
