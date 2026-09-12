"""Configuration for the topic modelling service."""

from __future__ import annotations

from dataclasses import dataclass

from settings import decimal, integer, required


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment.

    Connection settings are absent: database owns those. There is no stopword
    or token-length setting: the vocabulary is the content lemmas chunking
    stored, so grammar never reaches it.
    """

    num_topics: int
    passes: int
    random_state: int
    top_terms: int
    min_weight: float
    no_below: int
    no_above: float
    log_level: str

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            num_topics=integer("TOPIC_NUM_TOPICS"),
            passes=integer("TOPIC_PASSES"),
            random_state=integer("TOPIC_RANDOM_STATE"),
            top_terms=integer("TOPIC_TOP_TERMS"),
            min_weight=decimal("TOPIC_MIN_WEIGHT"),
            no_below=integer("TOPIC_NO_BELOW"),
            no_above=decimal("TOPIC_NO_ABOVE"),
            log_level=required("LOG_LEVEL").upper(),
        )
