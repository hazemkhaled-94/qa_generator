"""Command-line entry point for the review tool.

    python -m review.run --push questions
    python -m review.run --pull questions

Runs on the host against the same database the workers use, like `make
schema` and the bare stage targets. It is not in the backend image on
purpose: nothing in the pipeline calls it, it runs when a person decides
to review something, and adding the Argilla client to an image that
already carries torch and spaCy would be three more megabytes in six
containers for a tool none of them runs.
"""

from __future__ import annotations

import argparse
import logging
import sys

import telemetry
from database.qa_generator import engine
from question_generation.repository import QuestionCatalog
from review import datasets
from review.config import Settings
from review.records import ReviewRepository
from review.service import Catalogs, pull, push
from topic_modelling.repository import TopicCatalog

log = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    built = argparse.ArgumentParser(
        prog="python -m review.run",
        description="Put rows in front of a reviewer, and bring the "
        "verdicts back. The database decides; Argilla is where somebody "
        "reads.",
    )
    group = built.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--push",
        metavar="DATASET",
        choices=datasets.NAMES,
        help=f"send a sample for review: {', '.join(datasets.NAMES)}",
    )
    group.add_argument(
        "--pull",
        metavar="DATASET",
        choices=datasets.NAMES,
        help="bring back the verdicts somebody submitted",
    )
    group.add_argument(
        "--status", action="store_true", help="report how much has been reviewed"
    )
    return built


def main(argv: list[str] | None = None) -> int:
    """Runs the review command line.

    Args:
        argv: The arguments to parse, or None to read them from sys.argv.

    Returns:
        The process exit code.
    """
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("review")
    telemetry.trace_engine(engine())

    repository = ReviewRepository()

    if args.status:
        log.info("review: %s", repository.counts())
        return 0

    settings = Settings.load()
    catalogs = Catalogs(
        facts=repository, questions=QuestionCatalog(), topics=TopicCatalog()
    )

    if args.push:
        push(args.push, settings, catalogs)
    else:
        pull(args.pull, settings, catalogs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
