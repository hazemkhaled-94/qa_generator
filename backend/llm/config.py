"""Configuration for the served model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

from settings import Source, decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Which model to call, where, and how patiently.

    One set of values for every stage that calls a model: extraction reads
    passages with it and topic modelling names topics with it, and two
    settings for one served model is how the two come to disagree.

    A stage may name a different model with `overridden`, and only the
    model: the address, the mode and the patience are the deployment's, and
    the lease is derived from the last of those.
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
    #: Where a self-hosted runtime is, for a stage that overrides the model
    #: with one served there. None leaves every override on `base_url`.
    ollama_base_url: str | None = None

    @property
    def lease(self) -> timedelta:
        """How long a claim may go unfinished before a later run sweeps it.

        Derived rather than declared: a healthy worker may take the timeout
        on every attempt, and a lease below that fails a row still being
        worked. Doubled for the backoff between attempts and the checks after.
        """
        return timedelta(seconds=self.timeout_seconds * self.max_attempts * 2)

    def overridden(self, model: str | None) -> Settings:
        """These settings, calling a named model instead of the shared one.

        Returns self when nothing is named, so a caller needs no branch.

        **The address follows the provider.** litellm reads a model id's
        prefix to pick the provider, so a stage naming `ollama_chat/...`
        while `LLM_BASE_URL` points at Azure sends an Ollama request to
        Azure and is answered with a 404 - which is what happened the first
        time a phrasing model was pointed at a local runtime. The one thing
        a stage may override is the model, and an address that contradicts
        it is not a second override; it is the first one not working.

        Only for a self-hosted prefix, and only where `OLLAMA_BASE_URL`
        says where. Everything else keeps the shared address, because a
        hosted provider's own default is what `base_url` being absent
        means.

        Args:
            model: The model to call, or None to keep the shared one.
        """
        if not model:
            return self
        if model.startswith("ollama") and self.ollama_base_url:
            return replace(self, model=model, base_url=self.ollama_base_url)
        return replace(self, model=model)

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.
        """
        return cls(
            model=required("LLM_MODEL", source),
            base_url=optional("LLM_BASE_URL", source),
            structured_mode=required("LLM_STRUCTURED_MODE", source),
            temperature=decimal("LLM_TEMPERATURE", source),
            timeout_seconds=decimal("LLM_TIMEOUT_SECONDS", source),
            max_attempts=integer("LLM_MAX_ATTEMPTS", source),
            num_ctx=int(window)
            if (window := optional("LLM_NUM_CTX", source))
            else None,
            reasoning_effort=optional("LLM_REASONING_EFFORT", source),
            ollama_base_url=optional("OLLAMA_BASE_URL", source),
        )
