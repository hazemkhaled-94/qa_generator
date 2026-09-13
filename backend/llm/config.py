"""Configuration for the served model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from settings import decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Which model to call, where, and how patiently.

    One set of values for every stage that calls a model: extraction reads
    passages with it and topic modelling names topics with it, and two
    settings for one served model is how the two come to disagree.
    """

    model: str
    base_url: str | None
    structured_mode: str
    temperature: float
    timeout_seconds: float
    max_attempts: int

    @property
    def lease(self) -> timedelta:
        """How long a claim may go unfinished before a later run sweeps it.

        Derived rather than declared: a healthy worker may take the timeout
        on every attempt, and a lease below that fails a row still being
        worked. Doubled for the backoff between attempts and the checks after.
        """
        return timedelta(seconds=self.timeout_seconds * self.max_attempts * 2)

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            model=required("LLM_MODEL"),
            base_url=optional("LLM_BASE_URL"),
            structured_mode=required("LLM_STRUCTURED_MODE"),
            temperature=decimal("LLM_TEMPERATURE"),
            timeout_seconds=decimal("LLM_TIMEOUT_SECONDS"),
            max_attempts=integer("LLM_MAX_ATTEMPTS"),
        )
