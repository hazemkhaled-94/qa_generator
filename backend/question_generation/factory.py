"""Wiring for the question generation service."""

from __future__ import annotations

import logging
from dataclasses import replace

from llm.client import Client
from llm.config import Settings as ModelSettings
from question_generation.config import Settings
from question_generation.embedding import Embedder
from question_generation.generation import QuestionWriter
from question_generation.repository import QuestionCatalog, QuestionQueue
from question_generation.service import QuestionGenerationService
from question_generation.verification import QuestionChecker, Verifier

log = logging.getLogger(__name__)


def lease(settings: Settings, model: ModelSettings):
    """How long one topic may go unfinished before a run sweeps it.

    A topic is not a passage. It costs `per_topic` candidates, each of them
    a writer call and a verifier call, so the lease extraction derives for
    one call would fail a worker that is only halfway through its first
    topic.
    """
    return settings.lease(model.timeout_seconds * model.max_attempts)


def build_service(
    settings: Settings, model: ModelSettings
) -> QuestionGenerationService:
    """Wires the service and its collaborators.

    The verifier is a second model, named by QUESTIONS_VERIFIER_MODEL and
    served wherever the writer is. A model marking its own work agrees with
    itself, so running without one is worth saying out loud rather than
    letting the numbers quietly flatter the writer.
    """
    verifier_model = model
    if settings.verifier_model:
        verifier_model = replace(model, model=settings.verifier_model)
    else:
        log.warning(
            "QUESTIONS_VERIFIER_MODEL is unset, so %s is verifying its own "
            "questions. A model marking its own work agrees with itself, and "
            "the recoverability gate is the one gate that matters most.",
            model.model,
        )

    catalog = QuestionCatalog()
    return QuestionGenerationService(
        repository=QuestionQueue(
            lease=lease(settings, model), kinds=settings.fact_kinds
        ),
        writer=QuestionWriter(Client(model)),
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
            judge_phrasing=bool(settings.verifier_model),
            bounds=settings.answer_chars,
            overlap=settings.answer_overlap,
            long_answer_chars=settings.long_answer_chars,
        ),
        settings=settings,
    )
