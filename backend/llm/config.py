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
    #: The context window to ask the runtime for, or None to take its default.
    #: A self-hosted runtime reserves the whole window as key-value cache
    #: before it reads anything, so a model advertising 131,072 tokens holds
    #: gigabytes for a prompt of two thousand.
    num_ctx: int | None
    #: How much a thinking model may think before answering, or None to leave
    #: it to the model. `low`, `medium` and `high` let it think; anything else
    #: turns thinking off, which is what a structured answer wants.
    reasoning_effort: str | None

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
            num_ctx=int(window) if (window := optional("LLM_NUM_CTX")) else None,
            reasoning_effort=optional("LLM_REASONING_EFFORT"),
        )
