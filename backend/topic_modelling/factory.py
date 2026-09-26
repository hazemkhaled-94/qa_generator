"""Wiring for the topic modelling service."""

from __future__ import annotations

import logging

from blob_store.s3 import ExportBucket, ModelsBucket
from llm.client import Client
from topic_modelling.config import Settings
from topic_modelling.labels import TopicLabeller
from topic_modelling.repository import TopicQueue
from topic_modelling.service import TopicModellingService
from topic_modelling.topics import TopicFitter

log = logging.getLogger(__name__)


def build_service(
    settings: Settings, version: str | None = None
) -> TopicModellingService:
    """Wires the service and its collaborators.

    The labeller is left out when no model is configured.

    Args:
        settings: The environment this deployment reads.
        version: The configuration these settings came from, recorded on
            every topic the fit stores.

    Returns:
        The service, ready to drain the fit queue.
    """
    labeller = None
    if settings.model is not None:
        labeller = TopicLabeller(Client(settings.model), settings.languages)
    else:
        log.warning("no model configured; topics will be named by their terms only")

    return TopicModellingService(
        repository=TopicQueue(version=version),
        export=ExportBucket(),
        models=ModelsBucket(),
        fitter=TopicFitter(
            num_topics=settings.num_topics,
            passes=settings.passes,
            random_state=settings.random_state,
            top_terms=settings.top_terms,
            min_weight=settings.min_weight,
            no_below=settings.no_below,
            no_above=settings.no_above,
            passages_per_topic=settings.passages_per_topic,
        ),
        labeller=labeller,
        min_fact_share=settings.label_min_fact_share,
    )
