"""Wiring for the extraction service."""

from __future__ import annotations

from extraction.config import Settings
from extraction.extractors import (
    BridgeExtractor,
    DigestExtractor,
    ExtractorRegistry,
    LlmExtractor,
    TableExtractor,
)
from extraction.repository import PassageQueue
from extraction.service import ExtractionService
from extraction.validation import FactChecker
from llm.client import Client
from llm.config import Settings as ModelSettings


def build_service(
    model: ModelSettings,
    settings: Settings | None = None,
    version: str | None = None,
) -> ExtractionService:
    """Wires the per-passage service and its collaborators.

    Args:
        model: Which model to call, where, and how patiently.
        settings: What extraction writes. Read from the environment when not
            given.
        version: The configuration these settings came from, recorded on
            every fact written. None records nothing.

    Returns:
        The service a worker drains the passage queue with.
    """
    settings = settings or Settings.load()
    client = Client(model.overridden(settings.model))
    digests = settings.digests
    cap = settings.atomic_cap
    return ExtractionService(
        repository=PassageQueue(lease=model.lease, version=version),
        extractors=ExtractorRegistry(
            extractors=(TableExtractor(),), default=LlmExtractor(client, cap)
        ),
        checker=FactChecker(settings.digest_share),
        digest=DigestExtractor(client, digests) if digests else None,
        # Asked for in the prompt and enforced again on what comes back: a
        # model told to write four still writes nine, and a cap nothing
        # checks is a suggestion.
        atomic_cap=cap,
    )


def build_bridge(
    model: ModelSettings, settings: Settings | None = None
) -> BridgeExtractor:
    """Builds the reader of a group of passages.

    Args:
        model: Which model to call, where, and how patiently.
        settings: What extraction writes, for the model it names. Read from
            the environment when not given.

    Returns:
        The extractor the bridge pass calls once per group.
    """
    settings = settings or Settings.load()
    return BridgeExtractor(Client(model.overridden(settings.model)))
