"""Wiring for the question generation service."""

from __future__ import annotations

import logging

from llm.client import Client
from llm.config import Settings as ModelSettings
from nlp.embedding import Embedder
from nlp.entailment import Entailment
from nlp.qa import Extractive
from question_generation.catalog import QuestionCatalog
from question_generation.checker import QuestionChecker
from question_generation.config import Settings
from question_generation.generation import QuestionWriter
from question_generation.phrasing import PhrasingJudge
from question_generation.queue import QuestionQueue
from question_generation.service import QuestionGenerationService
from question_generation.verifier import Verifier

log = logging.getLogger(__name__)


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

    # Its own dial, defaulting to the verifier's model rather than to the
    # writer's: what it judges is still the writer's work, so the same
    # independence argument applies. It never sees a passage, which is what
    # makes a much smaller model reasonable here.
    phrasing_model = verifier_model.overridden(settings.phrasing_model)

    catalog = QuestionCatalog()
    return QuestionGenerationService(
        repository=QuestionQueue(
            kinds=settings.fact_kinds,
            # Recorded on every question written.
            version=version,
            # A passage that repeats across the corpus is its furniture,
            # and no gate downstream refuses a question about furniture.
            boilerplate_cosine=settings.boilerplate_cosine,
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
            explanation_chars=settings.explanation_chars,
            overlap=settings.answer_overlap,
            coverage=settings.answer_coverage,
            long_answer_chars=settings.long_answer_chars,
            elsewhere=catalog.elsewhere,
            elsewhere_passages=settings.elsewhere_passages,
            off_topic_overlap=settings.off_topic_overlap,
            phrasing=PhrasingJudge(Client(phrasing_model)),
            # Nothing is loaded until the pass first runs, so a worker whose
            # questions all pass recall never pays for the weights.
            entailment=(
                Entailment(settings.entailment_model, settings.encoder_max_tokens)
                if settings.entailment_model
                else None
            ),
            entailment_threshold=settings.entailment_threshold,
            about_overlap=settings.entailment_overlap,
            extractive=(
                Extractive(settings.answer_model, settings.encoder_max_tokens)
                if settings.answer_model
                else None
            ),
            extractive_confidence=settings.answer_confidence,
        ),
        settings=settings,
    )
