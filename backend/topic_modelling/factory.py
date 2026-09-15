"""Wiring for the topic modelling service."""

from __future__ import annotations

import logging

from blob_store.seaweedfs import ExportBucket
from llm.client import Client
from topic_modelling.config import Settings
from topic_modelling.labels import TopicLabeller
from topic_modelling.repository import TopicQueue
from topic_modelling.service import TopicModellingService
from topic_modelling.topics import TopicFitter

log = logging.getLogger(__name__)


def build_service(settings: Settings) -> TopicModellingService:
    """Wires the service and its collaborators.

    The labeller is left out when no model is configured.

    Args:
        settings: The environment this deployment reads.

    Returns:
        The service, ready to drain the fit queue.
    """
    labeller = None
    if settings.model is not None:
        labeller = TopicLabeller(Client(settings.model), settings.languages)
    else:
        log.warning("no model configured; topics will be named by their terms only")

    return TopicModellingService(
        repository=TopicQueue(),
        export=ExportBucket(),
        fitter=TopicFitter(
            num_topics=settings.num_topics,
            passes=settings.passes,
            random_state=settings.random_state,
            top_terms=settings.top_terms,
            min_weight=settings.min_weight,
            no_below=settings.no_below,
            no_above=settings.no_above,
        ),
        labeller=labeller,
    )
