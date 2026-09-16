"""One extractor per kind of passage, dispatched by block type.

Importing this loads litellm. Nothing outside the worker should name it.
"""

from __future__ import annotations

from extraction.extractors.base import ExtractionFailed, Extractor
from extraction.extractors.bridge import BridgeExtractor
from extraction.extractors.digest import DigestExtractor
from extraction.extractors.llm import LlmExtractor
from extraction.extractors.table import TableExtractor

__all__ = [
    "BridgeExtractor",
    "DigestExtractor",
    "ExtractionFailed",
    "Extractor",
    "ExtractorRegistry",
    "LlmExtractor",
    "TableExtractor",
]


class ExtractorRegistry:
    """Chooses the extractor for a passage's block type.

    Has a default: most block types are prose, so only the kinds readable
    without a model are claimed by name.
    """

    def __init__(self, extractors: tuple[Extractor, ...], default: Extractor) -> None:
        """Initialises the registry.

        Args:
            extractors: The readers claiming a block type by name.
            default: The reader everything unclaimed falls to.

        Raises:
            ValueError: If two extractors claim one block type.
        """
        self._default = default
        self._by_block_type: dict[str, Extractor] = {}
        for extractor in extractors:
            for block_type in extractor.block_types:
                if block_type in self._by_block_type:
                    raise ValueError(f"two extractors claim {block_type}")
                self._by_block_type[block_type] = extractor

    def for_block_type(self, block_type: str | None) -> Extractor:
        """Returns the extractor that reads a block type.

        Args:
            block_type: The parser's label for the passage, or None.

        Returns:
            The reader claiming it, or the default.
        """
        return self._by_block_type.get(block_type or "", self._default)

    @property
    def block_types(self) -> tuple[str, ...]:
        """Every block type claimed by name."""
        return tuple(sorted(self._by_block_type))
