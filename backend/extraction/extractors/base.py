"""Shared behaviour for every fact extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from extraction.models import CandidateFact, PassageToExtract, Provenance


class ExtractionFailed(Exception):
    """Raised when an extractor could not read a passage at all.

    Distinct from finding nothing, which returns an empty list.
    """


class Extractor(ABC):
    """Draws statements out of one passage.

    Every extractor takes the same passage and returns the same candidates,
    so the service routes without knowing which kind it holds. None of them
    judges its own output; that is the checker's job.
    """

    #: Block types this extractor claims. Empty means it claims none by name
    #: and can only be the fallback. A ClassVar rather than an abstract
    #: property, because a subclass would answer a property with a ClassVar
    #: anyway - which shadows the descriptor rather than implementing it.
    block_types: ClassVar[tuple[str, ...]]

    #: How this extractor works: llm or deterministic.
    method: ClassVar[str]

    @property
    def provenance(self) -> Provenance:
        """What produced these facts, recorded on each one."""
        return Provenance()

    @abstractmethod
    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Reads the statements out of one passage.

        Args:
            passage: The passage to read.

        Returns:
            One candidate per statement. Empty when it carries none.

        Raises:
            ExtractionFailed: If the passage could not be read at all.
        """
