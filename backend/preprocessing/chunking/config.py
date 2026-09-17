"""Configuration for the chunking service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import Source, boolean, integer, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Attributes:
        embedding_model: The model whose tokenizer sizes a passage.
        max_tokens: The token budget a passage is cut to.
        merge_peers: Whether to combine undersized neighbours that share a
            heading.
    """

    embedding_model: str
    max_tokens: int
    merge_peers: bool

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads the settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.

        Returns:
            The loaded settings.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number.
        """
        return cls(
            embedding_model=required("EMBEDDING_MODEL", source),
            max_tokens=integer("EMBEDDING_MAX_TOKENS", source),
            merge_peers=boolean("CHUNKING_MERGE_PEERS", source),
        )
