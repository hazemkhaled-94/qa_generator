"""Configuration for the ingestion service."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from settings import Source, csv, integer


@lru_cache(maxsize=1)
def pipeline_version() -> str:
    """The version this build of the pipeline stamps on what it stores.

    Read from `pyproject.toml` rather than from the environment, which is
    the one exception to how every other setting is loaded and the reason
    for it is what a version IS. It describes the build, not the
    deployment: two deployments of one image produce the same thing and
    should say so, and a deployment given the ability to name its own
    version can only use it to lie. It was an env var, and an env var is a
    second copy of a number that already exists in `pyproject.toml`.

    Walks up from this file rather than taking a fixed path: the repository
    has `backend/ingestion/config.py` under the root, and the image has
    `ingestion/config.py` under `/app` with `pyproject.toml` beside it.

    Raises:
        FileNotFoundError: If no `pyproject.toml` sits above this file,
            which means the image was built without it.
    """
    for folder in Path(__file__).resolve().parents:
        found = folder / "pyproject.toml"
        if found.is_file():
            return tomllib.loads(found.read_text(encoding="utf-8"))["project"][
                "version"
            ]
    raise FileNotFoundError(
        "no pyproject.toml above ingestion/config.py: the version the "
        "pipeline stamps on a stored object is read from it, and the image "
        "has to carry it. See backend/api/Dockerfile."
    )


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Attributes:
        pipeline_version: Recorded in the stored object's metadata. Read
            from `pyproject.toml`, not from the environment.
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
    def load(cls, source: Source = None) -> Settings:
        """Reads the settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.

        Returns:
            The loaded settings.

        Raises:
            KeyError: If a required setting is missing.
            ValueError: If MAX_FILE_SIZE_MB is not an integer.
        """
        return cls(
            pipeline_version=pipeline_version(),
            max_file_size_mb=integer("MAX_FILE_SIZE_MB", source),
            allowed_mime_types=csv("ALLOWED_MIME_TYPES", source),
        )
