"""Naming a fitted topic with the model that reads the passages."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from llm.client import Client, ModelUnavailable
from topic_modelling.models import FittedTopic

log = logging.getLogger(__name__)

#: How much of each representative passage the model is shown.
_EXCERPT_CHARS = 400

_SYSTEM = """You name the subject a set of documents share.

You are given the terms a statistical model found to define one topic, and a
few excerpts the model placed in it. Answer with the name of the subject.

Rules:
- A short noun phrase, at most six words. Not a sentence.
- Name the SUBJECT, not the document type: "Warehouse safety", not
  "Procedural document about warehouse safety".
- Write it in the language named below, because the report it appears in is
  read in that language.
- Use the terms and excerpts only. If they share no subject, say exactly
  "Mixed" and nothing else - a wrong name is worse than none, because a
  coverage report is read by its names.
- No quotation marks, no trailing punctuation."""


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
        self, topic: FittedTopic, language: str, excerpts: list[str]
    ) -> str | None:
        """Names one topic.

        Args:
            topic: The topic, read for its top terms.
            language: ISO 639-1 code of the language to answer in.
            excerpts: Text of the passages the topic holds most strongly.

        Returns:
            The name, stripped of quotes and trailing punctuation, or None if
            the model was unreachable or found no shared subject.
        """
        try:
            answer = self._client.answer(
                system=_SYSTEM,
                user=self._prompt(topic, language, excerpts),
                shape=_Label,
            )
        except ModelUnavailable as exc:
            log.warning("could not name topic %d: %s", topic.topic_index, exc)
            return None
        # One strip over the whole set, so a full stop inside a closing quote
        # takes the quote with it.
        name = answer.label.strip("\"'. \n\t")
        if not name or name.casefold() == "mixed":
            return None
        return name

    def _prompt(self, topic: FittedTopic, language: str, excerpts: list[str]) -> str:
        """Builds the user message for one topic."""
        shown = "\n".join(
            f"- {excerpt[:_EXCERPT_CHARS].strip()}" for excerpt in excerpts
        )
        return (
            f"Language: {self._languages.get(language, language)}\n\n"
            f"Defining terms, strongest first:\n{', '.join(topic.top_terms)}\n\n"
            f"Excerpts this topic covers:\n{shown}"
        )
