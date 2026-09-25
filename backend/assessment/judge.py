"""Asking one model for one judgement, and what to do when it will not.

Through `llm.client`, which is the same client every other model call in
this pipeline goes through: one retry policy, one timeout, one structured
mode, one place the reasoning tags are stripped and one place the tokens
and the spend are recorded. A judge built on a second client would be a
second set of all of those, disagreeing with the first the day one moved.

`arize-phoenix-evals` is the obvious alternative and is already a
dependency, but only on the host - `evaluation/second_opinion.py` uses it,
and no image carries it. Reaching for it here would put a second LLM
client, a second rate limiter and a pandas dependency into a worker
container to ask a question the client already in that container can ask.
What is borrowed from that package instead is the part worth borrowing:
the metric names, the labels and the scores. See `templates.py`.

An unanswerable judgement is an ABSTENTION and never a refusal. A model
that could not be reached has not found a problem with the artefact, and
recording that as disapproval would fill the review queue with rows whose
only fault was that Ollama was restarting.
"""

from __future__ import annotations

import logging

from assessment.models import ArtifactToJudge, Judgement
from assessment.templates import PROMPT_VERSION, TEMPLATES, Template
from llm.client import Client, ModelUnavailable

log = logging.getLogger(__name__)


class JudgeUnavailable(Exception):
    """Raised when no metric of an artefact could be judged at all.

    The difference between this and an abstention is how much was lost. One
    metric of three missing leaves a partial verdict worth recording; none
    of them answering is a row that was never judged, and the queue must be
    able to tell those apart - the first is `assessed`, the second `failed`
    and retryable.
    """


class Judge:
    """One model, asked each metric an artefact kind declares."""

    def __init__(self, client: Client) -> None:
        """Initialises the judge with the model it asks."""
        self._client = client

    @property
    def model(self) -> str:
        """The model answering, recorded beside what it answered."""
        return self._client.model

    def judge(self, artifact: ArtifactToJudge) -> tuple[Judgement, ...]:
        """Asks every metric this artefact's kind declares.

        One call per metric, deliberately. Three judgements in one call is
        what `question_generation/phrasing.py` exists to undo: over nineteen
        labelled cases a judgement sharing a call with a harder task was
        answered at chance in German while the harder task was answered
        well in both languages.

        Returns:
            One judgement per metric that answered, in template order. Short
            of the full set where a metric abstained.

        Raises:
            JudgeUnavailable: If no metric answered, which means the model
                is down rather than that the artefact is bad.
        """
        asked = TEMPLATES[artifact.kind]
        answered = [
            judged
            for template in asked
            if (judged := self._one(template, artifact)) is not None
        ]
        if not answered:
            raise JudgeUnavailable(
                f"{self.model} answered none of the {len(asked)} judgement(s) "
                f"about {artifact.kind} {artifact.artifact_id}"
            )
        if len(answered) < len(asked):
            log.warning(
                "%s %d: %d of %d judgement(s) answered; the rest abstained",
                artifact.kind,
                artifact.artifact_id,
                len(answered),
                len(asked),
            )
        return tuple(answered)

    def _one(self, template: Template, artifact: ArtifactToJudge) -> Judgement | None:
        """One metric, or None where the model could not be reached."""
        try:
            said = self._client.answer(
                system=template.system,
                user=template.rendered(**artifact.fields),
                shape=template.shape(),
                prompt_version=PROMPT_VERSION,
            )
        except ModelUnavailable as exc:
            # Logged and abstained, never raised past the artefact: losing
            # one opinion must not fail a row the other two judged.
            log.warning(
                "no %s judgement for %s %d: %s",
                template.metric,
                artifact.kind,
                artifact.artifact_id,
                exc,
            )
            return None

        label = str(said.label)  # pyright: ignore[reportAttributeAccessIssue]
        return Judgement(
            metric=template.metric,
            label=label,
            score=template.scores[label],
            approved=label == template.good,
            explanation=" ".join(
                str(said.explanation).split()  # pyright: ignore[reportAttributeAccessIssue]
            ),
        )
