"""The model-backed extractor for what a passage is about."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from database.qa_generator import FactKind
from extraction.extractors.base import ExtractionFailed, Extractor
from extraction.models import BULLET, CandidateFact, PassageToExtract, Provenance
from llm.client import Client, ModelUnavailable

#: Recorded on every summary and outline drawn with the prompt below.
PROMPT_VERSION = "1"

_SYSTEM = """You condense an excerpt into something a reader can read instead
of it.

You return two things about the same excerpt:

- `summary`: two or three sentences of prose saying what the excerpt is
  about and what it establishes. Someone who reads only this knows what the
  excerpt covers and what it decided.
- `outline`: the separate points the excerpt makes, one short line each,
  between two and six of them. Not sentences from the excerpt: the points
  behind them.

Rules, all of them mandatory:
- ADD NOTHING. Every number, date, name and amount you write must appear in
  the excerpt. Do not round, convert, infer or complete.
- Be shorter than the excerpt. A summary as long as what it summarises is of
  no use to anybody.
- Name the subject. Do not open with "it", "this" or "they".
- Write in the language of the excerpt.
- Keep the qualifiers that make a point true: the date, the place, the
  party, the unit, the condition.
- Say what the excerpt says, not what it implies and not what you know.

Worked example. Excerpt, under the context "Support > Response times":

  Standard requests are answered within 48 hours on working days. Urgent
  requests are answered within 4 hours and may be raised by phone.

  summary: "Support response times differ by urgency. Standard requests are
            answered within 48 hours counting working days only, and urgent
            requests within 4 hours. Urgent requests may also be raised by
            phone."
  outline: ["Standard requests: answered within 48 hours",
            "The 48-hour time counts working days only",
            "Urgent requests: answered within 4 hours",
            "Urgent requests may be raised by phone"]
"""


class _Digest(BaseModel):
    """The model's answer for one passage."""

    summary: str = Field(
        default="",
        description="Two or three sentences of prose saying what the excerpt "
        "is about and what it establishes.",
    )
    outline: list[str] = Field(
        default_factory=list,
        description="The separate points the excerpt makes, one short line "
        "each, between two and six of them.",
    )


class DigestExtractor(Extractor):
    """Reads a passage's summary and outline with a local model.

    One call produces both: they are the same reading of the same passage,
    and asking twice costs twice.
    """

    block_types: ClassVar[tuple[str, ...]] = ()
    method: ClassVar[str] = "llm"

    def __init__(self, client: Client, kinds: tuple[str, ...]) -> None:
        """Initialises the extractor.

        Args:
            client: The model it asks.
            kinds: Which of the summary and the outline to keep.
        """
        self._client = client
        self._kinds = kinds

    @property
    def provenance(self) -> Provenance:
        """What produced these facts, recorded on each one."""
        return Provenance(
            model=self._client.model,
            prompt_version=PROMPT_VERSION,
            temperature=self._client.temperature,
        )

    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Asks the model what one passage is about.

        Args:
            passage: The passage to condense.

        Returns:
            One candidate per kind this extractor was built for, citing every
            sentence of the passage. Empty when the model returned neither.

        Raises:
            ExtractionFailed: If the model could not be reached.
        """
        try:
            answer = self._client.answer(
                system=_SYSTEM, user=self._prompt(passage), shape=_Digest
            )
        except ModelUnavailable as exc:
            raise ExtractionFailed(str(exc)) from exc

        cited = tuple(sentence.index for sentence in passage.sentences)
        written: dict[str, str] = {
            FactKind.SUMMARY: answer.summary.strip(),
            FactKind.OUTLINE: _bulleted(answer.outline),
        }
        return [
            CandidateFact(statement=written[kind], sentences=cited, kind=kind)
            for kind in self._kinds
            if written[kind]
        ]

    @staticmethod
    def _prompt(passage: PassageToExtract) -> str:
        """Builds the user message: the heading trail and the passage text."""
        heading = (
            f"Context (not part of the excerpt, use it only to name what the "
            f"excerpt is about): {passage.section_path}\n\n"
            if passage.section_path
            else ""
        )
        return f"{heading}Excerpt:\n{passage.text}"


def _bulleted(points: list[str]) -> str:
    """Joins the points into one statement, one bullet per line."""
    kept = [point.strip().lstrip("-•* ").strip() for point in points]
    return "\n".join(f"{BULLET}{point}" for point in kept if point)
