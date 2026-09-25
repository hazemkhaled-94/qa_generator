"""Wiring for the assessment service."""

from __future__ import annotations

import logging

from assessment.config import Settings, judged_kinds
from assessment.judge import Judge
from assessment.repository import AssessmentQueue
from assessment.service import AssessmentService
from llm.client import Client
from llm.config import Settings as ModelSettings

log = logging.getLogger(__name__)


def judge_model(settings: Settings, shared: ModelSettings) -> ModelSettings:
    """The model that judges, which is not the one that wrote.

    `ASSESSMENT_JUDGE_MODEL` names it. Absent, the shared model - and that
    absence is worth warning about rather than refusing, because a judge
    that is also the writer is a model marking its own work. The same
    argument `question_generation.factory` makes about the verifier, and it
    bites harder here: the whole value of this phase is that it is an
    opinion the pipeline did not already hold.
    """
    return shared.overridden(settings.judge_model)


def build_service(
    settings: Settings, model: ModelSettings, version: str | None = None
) -> AssessmentService:
    """Wires the service and its collaborators."""
    asked = judge_model(settings, model)
    if asked.model == model.model and not settings.judge_model:
        log.warning(
            "%s is judging the pipeline's own output, because "
            "ASSESSMENT_JUDGE_MODEL names nothing. A model marking its own "
            "work agrees with itself, and this phase exists to hold an "
            "opinion the pipeline does not already hold.",
            asked.model,
        )
    return AssessmentService(
        repository=AssessmentQueue(
            kinds=judged_kinds(settings),
            sample=settings.sample,
            version=version,
        ),
        judge=Judge(Client(asked)),
    )
