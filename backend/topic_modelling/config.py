"""Configuration for the topic modelling service."""

from __future__ import annotations

from dataclasses import dataclass

import llm.config
from settings import Source, csv, decimal, integer, optional


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment."""

    #: The model that names a topic, or None when none is configured. Already
    #: carries TOPIC_MODEL where one is set, so a caller reads this and needs
    #: to know nothing about the override.
    model: llm.config.Settings | None
    #: ISO 639-1 code to the language's name, for the naming prompt.
    languages: dict[str, str]
    num_topics: int
    #: How many passages one topic is worth. Turns num_topics into a ceiling
    #: and scales the count to each language's share of the corpus. 0 fits
    #: num_topics whatever the corpus is.
    passages_per_topic: int
    passes: int
    random_state: int
    top_terms: int
    min_weight: float
    no_below: int
    no_above: float
    #: The smallest share of a topic's passages that must carry a validated
    #: fact before the topic is worth naming. Below it the topic is left
    #: unnamed without a model call. 0 names every topic.
    label_min_fact_share: float

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.
        """
        return cls(
            # TOPIC_MODEL names the model that labels a topic, where naming
            # the topics is worth a different one from reading the passages.
            # Absent keeps the shared one.
            model=llm.config.Settings.load(source).overridden(
                optional("TOPIC_MODEL", source)
            )
            if optional("LLM_MODEL", source)
            else None,
            languages=dict(
                pair.split(":", 1) for pair in csv("TOPIC_LANGUAGE_NAMES", source)
            ),
            num_topics=integer("TOPIC_NUM_TOPICS", source),
            passages_per_topic=integer("TOPIC_PASSAGES_PER_TOPIC", source),
            passes=integer("TOPIC_PASSES", source),
            random_state=integer("TOPIC_RANDOM_STATE", source),
            top_terms=integer("TOPIC_TOP_TERMS", source),
            min_weight=decimal("TOPIC_MIN_WEIGHT", source),
            no_below=integer("TOPIC_NO_BELOW", source),
            no_above=decimal("TOPIC_NO_ABOVE", source),
            label_min_fact_share=decimal("TOPIC_LABEL_MIN_FACT_SHARE", source),
        )
