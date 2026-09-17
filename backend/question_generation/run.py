"""Command-line entry point for the question generation service.

Every flag is the same operation as the route beside it under /questions.
Run with --help for the list.
"""

from __future__ import annotations

import logging
import sys

from llm.config import Settings as ModelSettings
from question_generation.config import Settings
from question_generation.factory import build_service, lease
from question_generation.repository import QuestionCatalog, QuestionQueue
from question_generation.service import balance, reverify
from stages.cli import queue_main

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Runs the question generation command line."""
    settings = Settings.load()
    model = ModelSettings.load()

    #: This stage's own operation. No model is called: it puts the stored
    #: questions through the gates that need none, which is how a fact
    #: re-judged or a citation deleted reaches the questions resting on it.
    extra = {
        "reverify": (
            "check every stored question again, without calling a model",
            "rejected",
            lambda within: reverify(QuestionCatalog(), settings, within),
        ),
        #: Which of the accepted questions make up the release. Accepting
        #: one says it is sound; this says what the SET looks like, and the
        #: two are different problems.
        "balance": (
            "draw a balanced release out of every accepted question",
            "released",
            lambda within: balance(QuestionCatalog(), settings, within),
        ),
    }

    def build():
        """Builds the service, naming the models it will call."""
        log.info(
            "writing with %s, verifying with %s, at %s",
            model.model,
            settings.verifier_model or model.model,
            model.base_url,
        )
        return build_service(settings, model)

    return queue_main(
        name="questions",
        module="question_generation.run",
        repository=lambda: QuestionQueue(lease=lease(settings, model)),
        build_service=build,
        argv=sys.argv[1:] if argv is None else argv,
        extra=extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
