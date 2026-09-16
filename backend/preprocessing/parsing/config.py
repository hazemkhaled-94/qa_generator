"""Configuration for the parsing service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import boolean, decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Attributes:
        ocr_char_threshold: Characters per page below which a document counts
            as scanned.
        min_confidence: Lowest confidence a conversion may carry and still be
            kept.
        table_mode: TableFormer mode, `fast` or `accurate`.
        heading_hierarchy: Whether the converter rebuilds the heading tree.
        document_timeout_seconds: Longest one conversion may run.
        artifacts_path: Where the converter's model weights are, or None to
            let it download them.
    """

    ocr_char_threshold: int
    min_confidence: float
    table_mode: str
    heading_hierarchy: bool
    document_timeout_seconds: float | None
    artifacts_path: str | None

    @classmethod
    def load(cls) -> Settings:
        """Reads the settings from the environment.

        Returns:
            The loaded settings.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number.
        """
        return cls(
            ocr_char_threshold=integer("PARSING_OCR_CHAR_THRESHOLD"),
            min_confidence=decimal("PARSING_MIN_CONFIDENCE"),
            table_mode=required("PARSING_TABLE_MODE"),
            heading_hierarchy=boolean("PARSING_HEADING_HIERARCHY"),
            document_timeout_seconds=decimal("PARSING_TIMEOUT_SECONDS"),
            artifacts_path=optional("DOCLING_ARTIFACTS_PATH"),
        )
