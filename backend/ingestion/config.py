"""Configuration for the ingestion service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import csv, integer, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Attributes:
        pipeline_version: Recorded in the stored object's metadata.
        max_file_size_mb: Largest upload accepted.
        allowed_mime_types: Media types the service stores.
    """

    pipeline_version: str
    max_file_size_mb: int
    allowed_mime_types: tuple[str, ...]

    @property
    def max_file_size_bytes(self) -> int:
        """The upload limit in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @classmethod
    def load(cls) -> Settings:
        """Reads the settings from the environment.

        Returns:
            The loaded settings.

        Raises:
            KeyError: If a required setting is missing.
            ValueError: If MAX_FILE_SIZE_MB is not an integer.
        """
        return cls(
            pipeline_version=required("PIPELINE_VERSION"),
            max_file_size_mb=integer("MAX_FILE_SIZE_MB"),
            allowed_mime_types=csv("ALLOWED_MIME_TYPES"),
        )
