"""Configuration for the question generation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from settings import decimal, integer, optional, required


@dataclass(frozen=True)
class Settings:
    """How many questions to write per topic, and what to hold them to.

    Which model writes them is not here: that is `llm.config.Settings`, which
    every stage that calls a model shares. Only the verifier's name is here,
    because it is the one model choice this stage makes on its own.
    """

    per_topic: int
    sample_size: int
    unanswerable_share: float
    duplicate_cosine: float
    embedding_model: str
    max_tokens: int
    verifier_model: str | None

    def lease(self, call_seconds: float) -> timedelta:
        """How long one topic may go unfinished before a run sweeps it.

        Derived from the work rather than declared. A topic is not a passage:
        it costs `per_topic` candidates and each one is two model calls, a
        writer's and a verifier's, so a lease sized for one call fails a
        worker that is only halfway through its first topic.

        Two calls per candidate and not a round number above it. This is
        already the worst case - every call taking its full timeout on every
        attempt - and padding a worst case is what makes a lease long enough
        to matter: a worker killed mid-topic leaves that row unclaimable
        until the lease runs out, and nothing but time moves it.
        """
        return timedelta(seconds=call_seconds * self.per_topic * 2)

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number.
        """
        return cls(
            per_topic=integer("QUESTIONS_PER_TOPIC"),
            sample_size=integer("QUESTIONS_FACT_SAMPLE"),
            unanswerable_share=decimal("QUESTIONS_UNANSWERABLE_SHARE"),
            duplicate_cosine=decimal("QUESTIONS_DUPLICATE_COSINE"),
            # The one embedding model, as chunking reads it: a question
            # embedded by a model other than the one a passage was sized by
            # measures distance in a space the corpus was never put in.
            embedding_model=required("EMBEDDING_MODEL"),
            max_tokens=integer("EMBEDDING_MAX_TOKENS"),
            # Absent means the writer verifies its own questions, which is
            # worth knowing about rather than guessing at, so the service
            # warns rather than failing.
            verifier_model=optional("QUESTIONS_VERIFIER_MODEL"),
        )
