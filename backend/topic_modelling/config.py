"""Configuration for the topic modelling service."""

from __future__ import annotations

from dataclasses import dataclass

import llm.config
from settings import csv, decimal, integer, optional


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment."""

    #: The model that names a topic, or None when none is configured.
    model: llm.config.Settings | None
    #: ISO 639-1 code to the language's name, for the naming prompt.
    languages: dict[str, str]
    num_topics: int
    passes: int
    random_state: int
    top_terms: int
    min_weight: float
    no_below: int
    no_above: float

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            model=llm.config.Settings.load() if optional("LLM_MODEL") else None,
            languages=dict(pair.split(":", 1) for pair in csv("TOPIC_LANGUAGE_NAMES")),
            num_topics=integer("TOPIC_NUM_TOPICS"),
            passes=integer("TOPIC_PASSES"),
            random_state=integer("TOPIC_RANDOM_STATE"),
            top_terms=integer("TOPIC_TOP_TERMS"),
            min_weight=decimal("TOPIC_MIN_WEIGHT"),
            no_below=integer("TOPIC_NO_BELOW"),
            no_above=decimal("TOPIC_NO_ABOVE"),
        )
