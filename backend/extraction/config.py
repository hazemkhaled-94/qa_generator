"""Configuration for the extraction service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from settings import decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Database and object-store connection settings are absent: those packages
    own theirs.
    """

    model: str
    model_base_url: str | None
    structured_mode: str
    temperature: float
    timeout_seconds: float
    max_attempts: int
    log_level: str

    @property
    def lease(self) -> timedelta:
        """How long a claim may go unfinished before a later run sweeps it.

        Derived rather than declared: a healthy worker may take the timeout
        on every attempt, and a lease below that fails a passage still being
        read. Doubled for the backoff between attempts and the checks after.
        """
        return timedelta(seconds=self.timeout_seconds * self.max_attempts * 2)

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            model=required("EXTRACTION_MODEL"),
            model_base_url=optional("EXTRACTION_MODEL_BASE_URL"),
            structured_mode=required("EXTRACTION_STRUCTURED_MODE"),
            temperature=decimal("EXTRACTION_TEMPERATURE"),
            timeout_seconds=decimal("EXTRACTION_TIMEOUT_SECONDS"),
            max_attempts=integer("EXTRACTION_MAX_ATTEMPTS"),
            log_level=required("LOG_LEVEL").upper(),
        )
