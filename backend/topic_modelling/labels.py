"""Naming a fitted topic with the model that reads the passages."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from llm.client import Client, ModelUnavailable
from topic_modelling.models import FittedTopic

log = logging.getLogger(__name__)

#: How much of each representative passage the model is shown.
_EXCERPT_CHARS = 400

#: Recorded on every topic named with the prompt below. Bumped whenever that
#: prompt changes what a name is.
PROMPT_VERSION = "1"

_SYSTEM = """ROLE
You are a cataloguer. You name the subject that a set of documents share, for
a coverage report a person reads by its names.

ACTION
Answer with the name of the subject the topic is about, or with "Mixed" when
it has none.

STEPS
1. Read the defining terms, strongest first.
2. Count how many of them name PARTS OF A DOCUMENT rather than a subject.
   If more than half do, answer "Mixed" and stop.
3. Read the excerpts and work out what they are all about.
4. If the terms and excerpts share no subject, answer "Mixed" and stop.
5. Check the names already taken. If the name you would give is one of them,
   name what separates this topic from that one instead; if nothing does,
   answer "Mixed".
6. Write the name in the language you are asked for.

CONTEXT
You are given the terms a statistical model found to define one topic, a few
excerpts the model placed in it, the language to answer in, and the names
already given to other topics of the same fit.

A wrong name is worse than none, because a coverage report is read by its
names. Two topics under one name cannot be told apart in that report.

EXAMPLES
  terms: shelving, pallet, forklift, aisle, load
    -> "Warehouse safety"
  terms: contents, foreword, appendix, index, shelving
    -> "Mixed"  (more than half name parts of a document)

FORMAT
Return `label`, and nothing else.

- A short noun phrase, at most six words. Not a sentence.
- Name the SUBJECT, not the document type: "Warehouse safety", not
  "Procedural document about warehouse safety".
- Write it in the language named in the message, because the report it
  appears in is read in that language.
- Use the terms and excerpts only. If they share no subject, say exactly
  "Mixed" and nothing else.
- ANSWER "Mixed" WHEN MORE THAN HALF THE TERMS NAME PARTS OF A DOCUMENT
  rather than a subject: contents, foreword, acknowledgements, copyright
  notice, revision history, release notes, appendix, index, glossary,
  version. One subject word among them does not make the topic that subject.
- DO NOT REUSE A NAME ALREADY TAKEN. If this topic is genuinely the one that
  name describes, name what separates it from that one instead; if nothing
  does, answer "Mixed".
- No quotation marks, no trailing punctuation."""

#: The user message, as the catalogue records it and `_prompt` renders it.
_USER = """Language: {{language}}

{{taken}}Defining terms, strongest first:
{{terms}}

Excerpts this topic covers:
{{excerpts}}"""


class _Label(BaseModel):
    """The name the model gives one topic."""

    label: str = Field(
        description="A short noun phrase naming the subject, at most six "
        "words, in the language asked for. 'Mixed' if the terms share none."
    )


class TopicLabeller:
    """Names a topic from its terms and a few of the passages it holds."""

    def __init__(self, client: Client, languages: dict[str, str]) -> None:
        """Initialises the labeller.

        Args:
            client: The served model.
            languages: ISO 639-1 code to the language's name, for the prompt.
        """
        self._client = client
        self._languages = languages

    @property
    def model(self) -> str:
        """The model that names a topic, recorded on it."""
        return self._client.model

    def label(
        self,
        topic: FittedTopic,
        language: str,
        excerpts: list[str],
        taken: Sequence[str] = (),
    ) -> str | None:
        """Names one topic.

        Args:
            topic: The topic, read for its top terms.
            language: ISO 639-1 code of the language to answer in.
            excerpts: Text of the passages the topic holds most strongly.
            taken: Names already given to other topics of this fit.

        Returns:
            The name, stripped of quotes and trailing punctuation, or None if
            the model was unreachable, found no shared subject, or repeated a
            name already taken.
        """
        try:
            answer = self._client.answer(
                system=_SYSTEM,
                user=self._prompt(topic, language, excerpts, taken),
                shape=_Label,
                prompt_version=PROMPT_VERSION,
            )
        except ModelUnavailable as exc:
            log.warning("could not name topic %d: %s", topic.topic_index, exc)
            return None
        # One strip over the whole set, so a full stop inside a closing quote
        # takes the quote with it.
        name = answer.label.strip("\"'. \n\t")
        if not name or name.casefold() == "mixed":
            return None
        # Asked for in the prompt and enforced here: an unnamed topic reads as
        # unnamed, where two topics under one name read as one topic.
        if any(name.casefold() == one.casefold() for one in taken):
            log.info("topic %d repeated the name %r", topic.topic_index, name)
            return None
        return name

    def _prompt(
        self,
        topic: FittedTopic,
        language: str,
        excerpts: list[str],
        taken: Sequence[str] = (),
    ) -> str:
        """Builds the user message for one topic."""
        shown = "\n".join(
            f"- {excerpt[:_EXCERPT_CHARS].strip()}" for excerpt in excerpts
        )
        already = f"Names already taken:\n{', '.join(taken)}\n\n" if taken else ""
        return (
            _USER.replace("{{language}}", self._languages.get(language, language))
            .replace("{{taken}}", already)
            .replace("{{terms}}", ", ".join(topic.top_terms))
            .replace("{{excerpts}}", shown)
        )
