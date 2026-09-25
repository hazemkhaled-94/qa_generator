"""Command-line entry point for the assessment phase.

Every flag is the same operation as the route beside it under /assessment.
Run with --help for the list.

The one thing this stage does that no other does is refuse to run at all:
`ASSESSMENT_ENABLED` off means the queue verbs move nothing and a drain
claims nothing, said once and clearly rather than by silently finding an
empty queue.
"""

from __future__ import annotations

import logging
import sys

from assessment.config import Settings, judged_kinds
from assessment.factory import build_service, judge_model
from assessment.repository import AssessmentQueue
from llm.check import before_work
from llm.config import Settings as ModelSettings
from settings.store import snapshot
from stages.cli import queue_main

log = logging.getLogger(__name__)

#: What is printed when the phase is switched off. A message rather than an
#: empty queue: a deployment that has not turned this on and cannot tell
#: the difference between "off" and "nothing to judge" will read the second
#: as a bug in the first.
OFF = (
    "ASSESSMENT_ENABLED is off, so nothing is judged. The pipeline is "
    "unaffected - this phase only ever records an opinion beside the "
    "checker's. Set ASSESSMENT_ENABLED=true in .env to run it."
)


def main(argv: list[str] | None = None) -> int:
    """Runs the assessment command line."""

    def configured() -> tuple[Settings, ModelSettings, str]:
        """Reads the settings as they stand, and what that configuration is.

        Called rather than captured, so every operation below answers to a
        value written since this process started.
        """
        source, version = snapshot()
        return Settings.load(source), ModelSettings.load(source), version

    # The flag alone, and deliberately not through `configured`: that one
    # also reads the model settings, which are required and have no
    # default. A deployment with this phase off and no LLM_MODEL set is a
    # perfectly ordinary deployment, and it must not be stopped at
    # start-up by a setting for a model nothing is going to call.
    if not Settings.load(snapshot()[0]).enabled:
        log.info("%s", OFF)
        return 0

    def build():
        """Builds the service, naming the model it will ask."""
        settings, model, version = configured()
        asked = judge_model(settings, model)
        log.info(
            "judging %s with %s",
            ", ".join(judged_kinds(settings)),
            asked.model,
        )
        return build_service(settings, model, version)

    def preflight() -> None:
        """Proves the judge answers before an artefact is claimed."""
        settings, model, _ = configured()
        before_work(judge_model(settings, model))

    def queue() -> AssessmentQueue:
        """The queue this stage claims from."""
        settings, _, _ = configured()
        return AssessmentQueue(
            kinds=judged_kinds(settings),
            sample=settings.sample,
        )

    #: Enrolling without queuing, which is what makes a dry run possible:
    #: it says how many artefacts this phase would cost a model call each
    #: before anybody commits to paying for them.
    extra = {
        "enrol": (
            (
                "create an assessment for every artefact that has none, "
                "without queuing any of them. What `--start` does first, "
                "on its own"
            ),
            "enrolled",
            lambda within: queue().enrol(),
        ),
    }

    return queue_main(
        name="assessment",
        module="assessment.run",
        repository=queue,
        build_service=build,
        argv=sys.argv[1:] if argv is None else argv,
        extra=extra,
        preflight=preflight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
