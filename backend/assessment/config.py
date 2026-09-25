"""Configuration for the assessment stage.

The two settings that matter are in `.env` rather than in
`configs/env/backend.env`, against this project's usual split. `.env` holds
credentials, ports and addresses; tuning goes in git. These are neither:
turning the phase on commits a deployment to a model call per artefact, and
naming the judge decides whose opinion is being recorded. Both are
deployment decisions that differ between a laptop and a server, which is
what `.env` is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from settings import Source, boolean, csv, integer, optional


@dataclass(frozen=True)
class Settings:
    """Whether to judge, what to judge, and who judges it."""

    #: Whether the phase runs at all. Off, the queue answers `/status` with
    #: an empty queue, `start` enrols nothing and the Dagster asset reports
    #: that it is off rather than that it did nothing.
    enabled: bool
    #: The model asked. Absent, the verifier's model, and absent that,
    #: LLM_MODEL - so a deployment that sets nothing still judges with
    #: something rather than refusing to start.
    judge_model: str | None
    #: Which artefacts are judged. Narrowing it is how a deployment pays for
    #: the questions and not for twenty thousand facts.
    kinds: tuple[str, ...]
    #: How many artefacts of each kind one `start` enrols, newest first.
    #: 0 is every one of them.
    sample: int

    @property
    def unlimited(self) -> bool:
        """Whether a sample size of 0 means every artefact."""
        return self.sample <= 0

    def lease(self, per_call: float) -> timedelta:
        """How long one artefact may go unfinished before a run sweeps it.

        Derived from what an artefact costs rather than configured: a
        question is three judgements and so three calls, and a lease sized
        for one would fail a worker halfway through a healthy row. Doubled
        for the backoff between attempts and the write after them.
        """
        return timedelta(seconds=per_call * _MOST_METRICS * 2)

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override."""
        return cls(
            enabled=boolean("ASSESSMENT_ENABLED", source),
            judge_model=optional("ASSESSMENT_JUDGE_MODEL", source),
            kinds=csv("ASSESSMENT_KINDS", source),
            sample=integer("ASSESSMENT_SAMPLE", source),
        )


#: The most judgements any one artefact costs, which is a question's three.
#: Read from the templates rather than written down, so adding a metric
#: lengthens the lease with it.
def _most_metrics() -> int:
    """How many calls the most expensive artefact kind costs."""
    from assessment.templates import TEMPLATES

    return max(len(one) for one in TEMPLATES.values())


_MOST_METRICS = _most_metrics()


def judged_kinds(settings: Settings) -> tuple[str, ...]:
    """The kinds this deployment judges, refusing a name that is not one.

    Raises:
        ValueError: If ASSESSMENT_KINDS names something that is not an
            artefact kind, naming both it and the three that are.
    """
    from assessment.templates import TEMPLATES

    unknown = [one for one in settings.kinds if one not in TEMPLATES]
    if unknown:
        raise ValueError(
            f"ASSESSMENT_KINDS names {', '.join(unknown)}, which "
            f"{'are' if len(unknown) > 1 else 'is'} not judged here. The "
            f"kinds are {', '.join(TEMPLATES)}."
        )
    return settings.kinds
