"""The model-backed extractor for claims no single passage states."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from database.qa_generator import FactKind
from extraction.extractors.base import ExtractionFailed
from extraction.models import CandidateFact, PassageToExtract, Provenance
from llm.client import Client, ModelUnavailable

#: Recorded on every bridge drawn with the prompt below.
PROMPT_VERSION = "1"

_SYSTEM = """You are shown several numbered excerpts that the corpus itself
says are about the same subject. You write the claims that need MORE THAN ONE
of them.

A bridge is a claim a reader gets only by holding two excerpts together: the
same duty stated for two parties, the same limit given in two places, one
excerpt naming what another defines, two periods of the same series.

A fact has two parts:
- `passages` is the list of excerpt NUMBERS the claim rests on. It must name
  at least TWO. If a claim is true of one excerpt on its own, it is not a
  bridge and you do not write it.
- `statement` is WRITTEN BY YOU: one short sentence carrying ONE claim, which
  a reader who cannot see the excerpts understands on its own.

Rules, all of them mandatory:
- AT LEAST TWO EXCERPTS per fact. This is the whole point.
- ONE CLAIM per statement. Exactly one finite verb.
- ADD NOTHING. Every number, date, name and amount in your statement must
  appear in one of the excerpts you name. Do NOT add up, average, subtract,
  convert or otherwise compute a value: a total nobody wrote down is not in
  the material. Relate what is written; do not calculate it.
- NAME THE SUBJECT. Never write "it", "this", "they", "he", "she".
- Write the statement in the language of the excerpts.
- Excerpts that merely sit near each other bridge nothing. Returning no facts
  is a correct answer and is better than a forced one.

Worked example. Two excerpts:

  [P0] Standard requests are answered within 48 hours on working days.
  [P1] Urgent requests are answered within 4 hours and may be raised by
       phone.

  RIGHT  passages: [0, 1]
         statement: "Support response times are stated separately for
                     standard and urgent requests."

  WRONG  passages: [0]
         statement: "A standard support request is answered within 48 hours."
         (one excerpt: an ordinary fact, not a bridge)

  WRONG  passages: [0, 1]
         statement: "A standard request is answered within 48 hours and an
                     urgent one within 4 hours."
         (two claims in one sentence: split it, or write the one above)

  WRONG  passages: [0, 1]
         statement: "The two response times differ by 44 hours."
         (44 appears in neither excerpt: that is a calculation, not a claim)
"""


class _Bridge(BaseModel):
    """One bridge as the model is asked to return it."""

    passages: list[int] = Field(
        description="The numbers of the excerpts this claim rests on. At least two."
    )
    statement: str = Field(
        description="ONE claim, written by you in one short sentence with "
        "exactly one finite verb, naming its subject and understandable "
        "without the excerpts."
    )


class _Bridges(BaseModel):
    """The model's answer for one group of passages."""

    facts: list[_Bridge] = Field(
        default_factory=list,
        description="Every claim needing more than one excerpt; may be empty.",
    )


class BridgeExtractor:
    """Reads claims that span a group of passages.

    Not an :class:`~extraction.extractors.base.Extractor`: the unit of work
    is a group of passages rather than one, so it is not routed by block type
    and is not in the registry.
    """

    method = "llm"

    def __init__(self, client: Client) -> None:
        """Initialises the extractor with the model it asks."""
        self._client = client

    @property
    def provenance(self) -> Provenance:
        """What produced these facts, recorded on each one."""
        return Provenance(
            model=self._client.model,
            prompt_version=PROMPT_VERSION,
            temperature=self._client.temperature,
        )

    def extract(self, offered: Sequence[PassageToExtract]) -> list[CandidateFact]:
        """Asks the model for the claims a group of passages shares.

        Args:
            offered: The passages to show, in the order a candidate names
                them by position.

        Returns:
            One candidate per claim, each naming the positions it rests on.
            Empty when fewer than two passages were offered.

        Raises:
            ExtractionFailed: If the model could not be reached.
        """
        if len(offered) < 2:
            return []
        try:
            answer = self._client.answer(
                system=_SYSTEM, user=self._prompt(offered), shape=_Bridges
            )
        except ModelUnavailable as exc:
            raise ExtractionFailed(str(exc)) from exc
        return [
            CandidateFact(
                statement=fact.statement,
                sentences=(),
                kind=FactKind.BRIDGE,
                passages=tuple(dict.fromkeys(fact.passages)),
            )
            for fact in answer.facts
            if fact.statement.strip() and fact.passages
        ]

    @staticmethod
    def _prompt(offered: Sequence[PassageToExtract]) -> str:
        """Builds the user message: one numbered block per passage."""
        blocks = []
        for position, passage in enumerate(offered):
            heading = f" ({passage.section_path})" if passage.section_path else ""
            blocks.append(f"[P{position}]{heading}\n{passage.text}")
        return "Excerpts:\n\n" + "\n\n".join(blocks)
