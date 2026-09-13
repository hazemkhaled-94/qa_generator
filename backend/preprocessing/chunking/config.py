"""Configuration for the chunking service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import boolean, integer, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Connection settings are absent: database and blob_store own theirs.
    """

    embedding_model: str
    max_tokens: int
    merge_peers: bool

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number.
        """
        return cls(
            embedding_model=required("EMBEDDING_MODEL"),
            max_tokens=integer("EMBEDDING_MAX_TOKENS"),
            merge_peers=boolean("CHUNKING_MERGE_PEERS"),
        )
