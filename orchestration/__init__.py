"""The code location Dagster loads: what may run, and when.

Three ways to set the pipeline going, and they are the three a person
already had:

    the whole corpus    `corpus` job, by hand or on the nightly schedule
    one stage           materialise that asset alone in the UI
    when work arrives   the `arrivals` sensor

None of them is required. Every stage still answers its own route, its own
make target and its own Start button, and this package being absent is a
pipeline that runs exactly as before with nobody deciding for it.

Loaded through configs/dagster/workspace.yaml.
"""

from __future__ import annotations

import logging

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    RunRequest,
    ScheduleDefinition,
    SkipReason,
    define_asset_job,
    sensor,
)

import telemetry
from orchestration import stages
from orchestration.client import Backend
from orchestration.settings import Settings

# The same logging and tracing every other process installs, so the
# orchestrator's lines land in the same index under the same field names
# and its spans join the trace the api continues.
telemetry.configure("orchestration")

log = logging.getLogger(__name__)

#: Everything, in order. Each asset waits for the stage before it to drain,
#: so this is the corpus going end to end in one run.
corpus = define_asset_job(
    name="corpus",
    selection=AssetSelection.all(),
    description="Every stage, in the order a document moves through them.",
)

#: A refit is corpus-wide and goes stale on every new document, so it is
#: the one thing worth a clock rather than a trigger. Off by default:
#: a schedule that starts running the moment the stack comes up is a
#: schedule nobody chose.
nightly = ScheduleDefinition(
    name="nightly_corpus",
    job=corpus,
    cron_schedule="0 2 * * *",
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.STOPPED,
    description="The whole pipeline at 02:00 UTC. Stopped until switched on.",
)


@sensor(
    job=corpus,
    minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    description="Starts the pipeline when a stage has work nobody has asked for.",
)
def arrivals():
    """Requests a run when the first stage has documents sitting `new`.

    This is the piece that makes the pipeline unattended, and it is worth
    being precise about what it does and does not change. A row still
    arrives `new` and no worker still looks at it; a stage still never
    sets another stage going. What this does is be the somebody who
    decides - the same decision the Start button makes, made on a timer
    by something watching for uploads.

    Watches parsing only. The stages after it are chained by the asset
    graph, so a second sensor per stage would race the run already
    working through them.
    """
    settings = Settings.load()
    queue = Backend(settings.backend_url).status("parsing")

    waiting = queue.rows.get("new", 0)
    if not waiting:
        return SkipReason("no document is waiting to be asked for")
    if not queue.idle:
        return SkipReason(
            f"{queue.outstanding} document(s) already queued or in progress"
        )

    log.info("%d document(s) uploaded and never asked for; starting", waiting)
    # Keyed on how many are waiting, so a second upload while a run is
    # working asks for another run rather than being folded into the one
    # already going - and two polls over an unchanged queue do not.
    return RunRequest(run_key=f"new-{waiting}")


defs = Definitions(
    assets=[
        stages.parsed_documents,
        stages.passages,
        stages.facts,
        stages.topics,
        stages.questions,
    ],
    asset_checks=[
        stages.parsed_documents_check,
        stages.passages_check,
        stages.facts_check,
        stages.topics_check,
        stages.questions_check,
    ],
    jobs=[corpus],
    schedules=[nightly],
    sensors=[arrivals],
)
