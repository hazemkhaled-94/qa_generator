"""Configuration for the parsing service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import boolean, decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Connection settings are absent: database and blob_store own theirs.
    """

    ocr_char_threshold: int
    table_mode: str
    heading_hierarchy: bool
    document_timeout_seconds: float | None
    artifacts_path: str | None
    log_level: str

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number.
        """
        return cls(
            ocr_char_threshold=integer("PARSING_OCR_CHAR_THRESHOLD"),
            table_mode=required("PARSING_TABLE_MODE"),
            heading_hierarchy=boolean("PARSING_HEADING_HIERARCHY"),
            document_timeout_seconds=decimal("PARSING_TIMEOUT_SECONDS"),
            artifacts_path=optional("DOCLING_ARTIFACTS_PATH"),
            log_level=required("LOG_LEVEL").upper(),
        )
