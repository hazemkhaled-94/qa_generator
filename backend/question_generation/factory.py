"""Wiring for the question generation service."""

from __future__ import annotations

import logging

from llm.client import Client
from llm.config import Settings as ModelSettings
from question_generation.catalog import QuestionCatalog
from question_generation.checker import QuestionChecker
from question_generation.config import Settings
from question_generation.embedding import Embedder
from question_generation.generation import QuestionWriter
from question_generation.queue import QuestionQueue
from question_generation.service import QuestionGenerationService
from question_generation.verifier import Verifier

log = logging.getLogger(__name__)


def lease(settings: Settings, model: ModelSettings):
    """How long one topic may go unfinished before a run sweeps it.

    A topic is not a passage. It costs `per_topic` candidates, each of them
    a writer call and a verifier call, so the lease extraction derives for
    one call would fail a worker that is only halfway through its first
    topic.
    """
    return settings.lease(model.timeout_seconds * model.max_attempts)


def models(
    settings: Settings, shared: ModelSettings
) -> tuple[ModelSettings, ModelSettings]:
    """The model that writes a question, and the model that checks it.

    QUESTIONS_MODEL and QUESTIONS_VERIFIER_MODEL name them; either absent is
    LLM_MODEL. A pair rather than a flag: what the gates need is whether the
    two came out different, not whether a setting was set.
    """
    return shared.overridden(settings.model), shared.overridden(settings.verifier_model)


def build_service(
    settings: Settings, model: ModelSettings, version: str | None = None
) -> QuestionGenerationService:
    """Wires the service and its collaborators.

    Two models, served wherever the shared one is: QUESTIONS_MODEL writes
    and QUESTIONS_VERIFIER_MODEL checks. A verifier that turns out to be the
    writer is warned about.
    """
    writer_model, verifier_model = models(settings, model)
    independent = verifier_model.model != writer_model.model
    if not independent:
        log.warning(
            "%s is verifying its own questions. A model marking its own work "
            "agrees with itself, and the recoverability gate is the one gate "
            "that matters most. Set QUESTIONS_VERIFIER_MODEL to a different "
            "model.",
            writer_model.model,
        )

    catalog = QuestionCatalog()
    return QuestionGenerationService(
        repository=QuestionQueue(
            lease=lease(settings, model),
            kinds=settings.fact_kinds,
            # Recorded on every question written.
            version=version,
        ),
        writer=QuestionWriter(Client(writer_model)),
        checker=QuestionChecker(
            embedder=Embedder(settings.embedding_model, settings.max_tokens),
            verifier=Verifier(Client(verifier_model)),
            nearest=catalog.nearest,
            threshold=settings.duplicate_cosine,
            # Only an independent model's opinion of a question may reject
            # it. A writer marking its own work rejected `According to the
            # ECB and NCAs, who conducts the due diligence check?` for
            # naming nothing, which is the kind of loss a gate must not
            # cause. Recoverability is unaffected: that one is checkable.
            judge_phrasing=independent,
            # For the same reason, and it matters more here: the entailment
            # pass can only accept, so a writer running it over its own
            # answer would wave through everything its own recall missed.
            entail=independent,
            bounds=settings.answer_chars,
            overlap=settings.answer_overlap,
            long_answer_chars=settings.long_answer_chars,
            elsewhere=catalog.elsewhere,
            elsewhere_passages=settings.elsewhere_passages,
            off_topic_overlap=settings.off_topic_overlap,
        ),
        settings=settings,
    )
