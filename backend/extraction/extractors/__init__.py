"""One extractor per kind of passage, dispatched by block type.

Importing this loads litellm. Nothing outside the worker should name it.
"""

from __future__ import annotations

from extraction.extractors.base import ExtractionFailed, Extractor
from extraction.extractors.llm import LlmExtractor
from extraction.extractors.table import TableExtractor

__all__ = [
    "ExtractionFailed",
    "Extractor",
    "ExtractorRegistry",
    "LlmExtractor",
    "TableExtractor",
]


class ExtractorRegistry:
    """Chooses the extractor for a passage's block type.

    Has a default, unlike the parsing registry: Docling's label set is
    open-ended and most of it is prose, so only the kinds readable without a
    model are claimed by name.
    """

    def __init__(self, extractors: tuple[Extractor, ...], default: Extractor) -> None:
        """Initialises the registry, refusing two claims on one block type."""
        self._default = default
        self._by_block_type: dict[str, Extractor] = {}
        for extractor in extractors:
            for block_type in extractor.block_types:
                if block_type in self._by_block_type:
                    raise ValueError(f"two extractors claim {block_type}")
                self._by_block_type[block_type] = extractor

    def for_block_type(self, block_type: str | None) -> Extractor:
        """Returns the extractor that reads a block type, or the default."""
        return self._by_block_type.get(block_type or "", self._default)

    @property
    def block_types(self) -> tuple[str, ...]:
        """Every block type claimed by name."""
        return tuple(sorted(self._by_block_type))
