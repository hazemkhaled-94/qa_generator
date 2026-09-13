"""Configuration for the ingestion service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import csv, integer, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Connection settings are absent on purpose: database and blob_store each
    own theirs, and this service builds neither an engine nor an S3 client.
    """

    pipeline_version: str
    max_file_size_mb: int
    allowed_mime_types: tuple[str, ...]

    @property
    def max_file_size_bytes(self) -> int:
        """The upload limit expressed in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment.

        Raises:
            ValueError: If MAX_FILE_SIZE_MB is not an integer.
        """
        return cls(
            pipeline_version=required("PIPELINE_VERSION"),
            max_file_size_mb=integer("MAX_FILE_SIZE_MB"),
            allowed_mime_types=csv("ALLOWED_MIME_TYPES"),
        )
