"""One asset per pipeline stage, chained in the order the corpus moves.

What an asset does here is decide that a stage should run, and then watch:
it posts `start` and polls `/status` until nothing claimable is left. The
work happens where it already happened, in the stage's worker container,
claimed one row at a time with FOR UPDATE SKIP LOCKED.

That is the whole design. Dagster is the fourth face onto a queue that
already had three, and the one the README reserved a slot for: "a stage
never sets another stage going - somebody decides it should run". The
orchestrator is a somebody. It is not a new execution engine, and moving
the stages into ops would have thrown away the lease sweep, the horizontal
scaling and a claim protocol that works, in exchange for putting a
473-second model call inside a Dagster op.

So an asset is cheap and the pipeline is unchanged by its absence. Nothing
here is required for the pipeline to run; it is required for it to run
unattended.
"""

# No `from __future__ import annotations` in this module, deliberately.
# Dagster resolves the type of an asset's `context` parameter when the
# decorator runs, and under that import every annotation is a string: the
# code location then fails to load with "Cannot annotate `context` with
# type AssetExecutionContext" while pointing at a parameter annotated with
# exactly that.

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    MetadataValue,
    Output,
    asset,
    asset_check,
)

from orchestration.client import Backend
from orchestration.settings import Settings
from telemetry import pipeline


def group(stage: str) -> str:
    """What Dagster files one stage's asset under.

    A group per stage rather than one `pipeline` group holding all seven.
    Dagster sorts both its asset list and its group list by name, and
    `assessments, facts, parsed_documents, passages, questions, topics` is
    not the order a corpus moves through them.

    The number is the stage's place in the pipeline and means what it means
    in Phoenix's project names and Argilla's dataset names. `stage_` in
    front because a Dagster group name must be a valid Python identifier
    and an identifier may not begin with a digit.
    """
    at = pipeline.POSITION.get(stage)
    return f"stage_{at}_{stage}" if at else stage


#: The stages, in the order a document moves through them, as
#: (asset name, route prefix, what its queue counts).
#:
#: `topics` is absent: it has no `start`, because a fit is asked for rather
#: than begun, so it gets an asset of its own below.
ROW_STAGES = (
    ("parsed_documents", "parsing", "document"),
    ("passages", "chunking", "document"),
    ("facts", "extraction", "passage"),
)


def _run(context: AssetExecutionContext, stage: str, unit: str) -> Output[int]:
    """Starts one stage, waits for it, and reports what moved.

    Shared by every row-based stage: they differ in their prefix and in
    what they count, and in nothing else, which is the same property the
    HTTP routes and the make targets are built on.
    """
    settings = Settings.load()
    backend = Backend(settings.backend_url)

    queued = backend.act(stage, "start")
    context.log.info("%s: %d %s(s) queued", stage, queued, unit)

    queue = backend.drain(
        stage, timeout=settings.drain_timeout, poll=settings.poll_seconds
    )
    done = sum(
        count for status, count in queue.rows.items() if status not in ("new", "failed")
    )
    return Output(
        done,
        metadata={
            "queued by this run": queued,
            f"{unit}s done": done,
            "failed": queue.failed,
            "queue": MetadataValue.json(queue.rows),
        },
    )


def _stage_asset(name: str, stage: str, unit: str, deps: list[str]):
    """Builds one stage's asset and the check that reads its failures."""

    @asset(
        name=name,
        deps=[AssetKey(one) for one in deps],
        group_name=group(stage),
        description=f"Runs {stage} over everything not asked for yet, and "
        f"waits for the workers to drain it.",
        compute_kind="queue",
    )
    def _materialise(context: AssetExecutionContext) -> Output[int]:
        """Starts the stage and waits."""
        return _run(context, stage, unit)

    @asset_check(
        asset=name,
        name="nothing_failed",
        description=f"Whether {stage} left any row failed.",
    )
    def _check() -> AssetCheckResult:
        """Reports the stage's failed rows.

        A check rather than an exception, because a failed row is not a
        failed run: the rest of the corpus went through, the reason is
        recorded against the row, and `retry` is what moves it. Raising
        here would stop the stage after it from ever running because one
        document of four hundred was a scanned image.
        """
        settings = Settings.load()
        queue = Backend(settings.backend_url).status(stage)
        return AssetCheckResult(
            passed=queue.failed == 0,
            metadata={
                "failed": queue.failed,
                "retry with": f"POST /{stage}/retry",
                "queue": MetadataValue.json(queue.rows),
            },
        )

    return _materialise, _check


_BUILT = [
    _stage_asset(name, stage, unit, [ROW_STAGES[index - 1][0]] if index else [])
    for index, (name, stage, unit) in enumerate(ROW_STAGES)
]

parsed_documents, parsed_documents_check = _BUILT[0]
passages, passages_check = _BUILT[1]
facts, facts_check = _BUILT[2]


@asset(
    deps=[AssetKey("facts")],
    group_name=group("topic_modelling"),
    description="Fits one topic model per language over the whole corpus, "
    "replacing every topic there was.",
    compute_kind="queue",
)
def topics(context: AssetExecutionContext) -> Output[int]:
    """Asks for a fit and waits for it.

    Topic modelling has no `start` and no per-item scope. Every topic is
    estimated jointly over one vocabulary, so no topic can be refitted on
    its own and a new document does not make one topic stale - it makes
    all of them stale. Asking is what creates the work.
    """
    settings = Settings.load()
    backend = Backend(settings.backend_url)

    fit = backend.discover()
    context.log.info("topics: fit %s asked for", fit)

    queue = backend.drain(
        "topics", timeout=settings.drain_timeout, poll=settings.poll_seconds
    )
    return Output(
        fit,
        metadata={
            "fit": fit,
            "failed": queue.failed,
            "queue": MetadataValue.json(queue.rows),
        },
    )


@asset_check(
    asset="topics", name="nothing_failed", description="Whether the fit failed."
)
def topics_check() -> AssetCheckResult:
    """Reports whether the fit succeeded.

    A fit that fails leaves the working topics in place with the reason
    beside them, so this failing means the corpus is being described by
    the topics from before rather than by none.
    """
    settings = Settings.load()
    queue = Backend(settings.backend_url).status("topics")
    return AssetCheckResult(
        passed=queue.failed == 0,
        metadata={"failed": queue.failed, "queue": MetadataValue.json(queue.rows)},
    )


@asset(
    deps=[AssetKey("topics")],
    group_name=group("question_generation"),
    description="Writes the test questions, one topic at a time.",
    compute_kind="queue",
)
def questions(context: AssetExecutionContext) -> Output[int]:
    """Starts question generation and waits.

    The slowest asset by a distance and the one most likely to hit the
    drain timeout: its unit is a topic, and a topic is QUESTIONS_PER_TOPIC
    candidates each costing a writer call and a verifier call.
    """
    return _run(context, "questions", "topic")


@asset_check(
    asset="questions",
    name="nothing_failed",
    description="Whether question generation left any topic failed.",
)
def questions_check() -> AssetCheckResult:
    """Reports the topics question generation could not write."""
    settings = Settings.load()
    queue = Backend(settings.backend_url).status("questions")
    return AssetCheckResult(
        passed=queue.failed == 0,
        metadata={
            "failed": queue.failed,
            "retry with": "POST /questions/retry",
            "queue": MetadataValue.json(queue.rows),
        },
    )


@asset(
    name="assessments",
    deps=[AssetKey("questions")],
    group_name=group("assessment"),
    description="The evaluation phase: an LLM judge over every fact, topic "
    "and question the pipeline produced. Records an opinion beside the "
    "checker's verdict and changes nothing.",
    compute_kind="queue",
)
def assessments(context: AssetExecutionContext) -> Output[int]:
    """Judges everything the pipeline produced, if the phase is switched on.

    Last in the graph, and that position is the whole point: the judge is
    asked about the corpus as it finally stands, not about a fact that a
    later revalidation would have rejected anyway.

    Switched off, this materialises rather than fails. A deployment that
    has not turned the phase on has not got a broken pipeline, and an asset
    that went red every night because a cost decision was made deliberately
    is an asset nobody would keep.
    """
    settings = Settings.load()
    backend = Backend(settings.backend_url)

    plan = backend.plan("assessment")
    if not plan.get("enabled"):
        context.log.info("assessment: ASSESSMENT_ENABLED is off; nothing judged")
        return Output(
            0,
            metadata={
                "enabled": False,
                "why": MetadataValue.text(
                    "ASSESSMENT_ENABLED is off in .env. The pipeline is "
                    "unaffected: this phase only records an opinion beside "
                    "the checker's verdict."
                ),
            },
        )

    queued = backend.act("assessment", "start")
    context.log.info("assessment: %d artifact(s) queued", queued)

    queue = backend.drain(
        "assessment", timeout=settings.drain_timeout, poll=settings.poll_seconds
    )
    done = produced(queue.rows)
    return Output(
        done,
        metadata={
            "enabled": True,
            "judge": plan.get("judge_model") or "LLM_MODEL (judging its own work)",
            "kinds judged": MetadataValue.json(plan.get("kinds") or []),
            "queued by this run": queued,
            "artifacts judged": done,
            "failed": queue.failed,
            "queue": MetadataValue.json(queue.rows),
        },
    )


@asset_check(
    asset="assessments",
    name="nothing_failed",
    description="Whether the judge could not be reached for any artefact.",
)
def assessments_check() -> AssetCheckResult:
    """Reports the artefacts the judge could not be asked about.

    A failure here is the model being unreachable, not an artefact being
    bad: a judgement that was refused is recorded as a refusal and is not
    a failed row.
    """
    settings = Settings.load()
    queue = Backend(settings.backend_url).status("assessment")
    return AssetCheckResult(
        passed=queue.failed == 0,
        metadata={
            "failed": queue.failed,
            "retry with": "POST /assessment/retry",
            "queue": MetadataValue.json(queue.rows),
        },
    )


@asset_check(
    asset="assessments",
    name="judge_agrees_with_the_checker",
    description="Where the judge refused something the pipeline kept.",
)
def assessments_agreement() -> AssetCheckResult:
    """Reports the disagreements, and never fails on them.

    A disagreement is not a defect. It is either a check that let something
    through or a judge that is wrong, and which of those it is cannot be
    decided here - `evaluation/README.md` records an LLM judge measured at
    chance on the German half of this corpus. So this check passes whatever
    it finds and carries the count, which is a number worth watching on the
    asset page and never a reason to fail a run.

    What acts on it is a person: `make review-push-questions`.
    """
    settings = Settings.load()
    quality = Backend(settings.backend_url).judged()
    judged = quality.get("judged", 0)
    approved = quality.get("approved", 0)
    disagreements = quality.get("disagreements", 0)
    return AssetCheckResult(
        passed=True,
        metadata={
            "judged": judged,
            "approved": approved,
            "approval rate": f"{approved / judged:.0%}" if judged else "nothing judged",
            "kept by the pipeline, refused by the judge": disagreements,
            "what to do": MetadataValue.text(
                "Neither side is right by default. Put them in front of "
                "somebody: make review-push-questions, then review-pull-questions."
            ),
            "by metric": MetadataValue.json(quality.get("metrics") or []),
        },
    )


#: Every stage that has an asset, as (asset name, route prefix, unit).
#: `ROW_STAGES` plus the three that are asked for rather than started.
ALL_STAGES = ROW_STAGES + (
    ("topics", "topics", "fit"),
    ("questions", "questions", "topic"),
    ("assessments", "assessment", "artifact"),
)

#: Queue statuses that mean a row has NOT been produced yet. Everything
#: else counts as done, which is what a materialisation reports.
_UNDONE = ("new", "pending", "in_progress", "failed")


def produced(rows: dict[str, int]) -> int:
    """How many rows of a stage's queue have been worked."""
    return sum(count for status, count in rows.items() if status not in _UNDONE)
