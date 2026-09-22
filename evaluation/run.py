"""Command-line entry point for the golden-set experiments.

    python -m evaluation.run --upload extraction-golden
    python -m evaluation.run --score  extraction-golden

Runs on the host against the served model and the containerised Phoenix,
like `make test-eval` and for the same reason: it calls a real model, so
it is never part of a gate.
"""

from __future__ import annotations

import argparse
import logging
import sys

import telemetry
from evaluation import experiments
from evaluation.config import Settings

log = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    built = argparse.ArgumentParser(
        prog="python -m evaluation.run",
        description="Put the golden cases in Phoenix and score a served "
        "model against them, so two runs can be compared by more than "
        "their scrollback.",
    )
    group = built.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--upload",
        metavar="DATASET",
        choices=experiments.NAMES,
        help=f"send the cases to Phoenix: {', '.join(experiments.NAMES)}",
    )
    group.add_argument(
        "--score",
        metavar="DATASET",
        choices=experiments.NAMES,
        help="run the model against an uploaded set and record the scores",
    )
    group.add_argument(
        "--second-opinion",
        metavar="RUN",
        dest="run_id",
        help="put one run's accepted answers to an independent judge and "
        "report where it disagrees with the gates. Never a verdict: the "
        "output is a queue for review.",
    )
    built.add_argument(
        "--limit",
        type=int,
        default=None,
        help="how many answers --second-opinion judges. One model call each.",
    )
    return built


def main(argv: list[str] | None = None) -> int:
    """Runs the evaluation command line.

    Args:
        argv: The arguments to parse, or None to read them from sys.argv.

    Returns:
        The process exit code.
    """
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("evaluation")

    settings = Settings.load()
    if args.run_id:
        return _second_opinion(args.run_id, args.limit)

    if args.upload:
        experiments.upload(args.upload, settings)
        return 0

    try:
        experiments.run(args.score, settings)
    except NotImplementedError as refused:
        # Not a crash: the set has no task on purpose, and the message
        # says what measures it instead.
        log.error("%s", refused)
        return 2
    return 0


def _second_opinion(run: str, limit: int | None) -> int:
    """Judges one run's accepted answers and prints the disagreements.

    Asks the VERIFIER's model rather than the writer's, for the reason the
    verifier is a second model at all: a model marking its own work agrees
    with itself.
    """
    from evaluation import second_opinion
    from llm.config import Settings as ModelSettings
    from question_generation.config import Settings as QuestionSettings

    model = ModelSettings.load()
    questions = QuestionSettings.load()
    asked = model.overridden(questions.verifier_model).model

    try:
        judged = second_opinion.judge(run, asked, limit or second_opinion.SAMPLE)
    except RuntimeError as empty:
        log.error("%s", empty)
        return 2

    for line in second_opinion.report(judged):
        log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
