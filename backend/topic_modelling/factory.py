"""Wiring for the topic modelling service."""

from __future__ import annotations

from topic_modelling.config import Settings
from topic_modelling.repository import TopicQueue
from topic_modelling.service import TopicModellingService
from topic_modelling.topics import TopicFitter


def build_service(settings: Settings) -> TopicModellingService:
    """Wires the service and its collaborators."""
    return TopicModellingService(
        repository=TopicQueue(),
        fitter=TopicFitter(
            num_topics=settings.num_topics,
            passes=settings.passes,
            random_state=settings.random_state,
            top_terms=settings.top_terms,
            min_weight=settings.min_weight,
            no_below=settings.no_below,
            no_above=settings.no_above,
        ),
    )
