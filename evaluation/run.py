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


if __name__ == "__main__":
    raise SystemExit(main())
