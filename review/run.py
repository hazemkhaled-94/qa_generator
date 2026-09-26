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
from question_generation.catalog import QuestionCatalog
from review import datasets, judged
from review.config import Settings
from review.records import ReviewRepository
from review.service import Catalogs, drop, pull, push
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
    group.add_argument(
        "--delete",
        action="store_true",
        help="delete every dataset in the workspace. What `make wipe` runs "
        "once the corpus is gone. Irreversible: pull first if a verdict "
        "nobody has brought back still matters.",
    )
    built.add_argument(
        "--all",
        action="store_true",
        dest="everything",
        help="push every row of that kind rather than a sample, so Argilla "
        "holds the corpus and a reviewer filters there. One dataset per "
        "kind either way.",
    )
    built.add_argument(
        "--ids",
        metavar="ID,ID",
        help="push exactly these question ids instead of a sample. What "
        "`make second-opinion` prints, so a disagreement between the gates "
        "and an independent judge reaches somebody who can settle it.",
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
        for name, counted in repository.counts().items():
            agreement = counted["agreement"]
            log.info(
                "%s: %d of %d reviewed%s%s",
                name,
                counted["reviewed"],
                counted["total"],
                (
                    f" ({', '.join(f'{v} {k}' for k, v in sorted(counted['verdicts'].items()))})"
                    if counted["verdicts"]
                    else ""
                ),
                (
                    f"; agreed with the model on {counted['agreed']} "
                    f"of them, {agreement:.0%}"
                    if agreement is not None
                    else "; nobody has looked, so there is no agreement to report"
                ),
            )
        return 0

    settings = Settings.load()

    # Before the catalogs, which read a database this does not need: a
    # wipe deletes the corpus first, and asking Argilla to forget it must
    # not depend on the rows still being there.
    if args.delete:
        log.info("deleted %d dataset(s)", drop(settings))
        return 0

    catalogs = Catalogs(
        facts=repository,
        questions=QuestionCatalog(),
        topics=TopicCatalog(),
        # What the evaluation phase said about each row, shown beside the
        # checker's verdict. Wired unconditionally: a deployment that has
        # never run the phase has no assessment rows, so every record
        # carries `not_judged` and nothing has to know whether it is on.
        judge=judged.catalog(),
    )

    if args.push:
        push(
            args.push,
            settings,
            catalogs,
            _ids(args.ids, args.push),
            everything=args.everything,
        )
    else:
        pull(args.pull, settings, catalogs)
    return 0


def _ids(named: str | None, dataset: str) -> list[int] | None:
    """The question ids a command named, or None for a sample.

    Raises:
        SystemExit: If they are not numbers, or the set has no queue behind
            it. A typo here would otherwise push a sample and look like it
            worked.
    """
    if not named:
        return None
    if dataset != datasets.QUESTIONS:
        raise SystemExit(
            f"--ids names questions to push and {dataset} is not that set; "
            f"the other two are sampled."
        )
    try:
        return [int(one) for one in named.replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"--ids takes question ids; {named!r} is not a list") from None


if __name__ == "__main__":
    raise SystemExit(main())
