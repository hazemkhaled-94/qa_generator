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

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dagster import (
    AssetCheckEvaluation,
    AssetKey,
    AssetMaterialization,
    AssetSelection,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    MetadataValue,
    RunRequest,
    ScheduleDefinition,
    SensorResult,
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
def arrivals(context):
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

    The run key counts the decisions rather than the documents, and the
    cursor is what makes that possible. Dagster remembers a run key for as
    long as it keeps the run, so a key naming only the queue depth is a key
    that comes round again: three documents uploaded, run, drained, three
    more uploaded is `new-3` twice, and the second one launches nothing at
    all. The counter moves when the queue this sensor is looking at differs
    from the one it last asked about, so two polls over an unchanged queue
    still share a key and still ask for one run.
    """
    settings = Settings.load()
    queue = Backend(settings.backend_url).status("parsing")

    asked = json.loads(context.cursor) if context.cursor else {}
    waiting = queue.rows.get("new", 0)
    if not waiting:
        # Recorded, not just skipped: an emptied queue is what tells the
        # next arrival that it is a new one rather than the same one again.
        context.update_cursor(json.dumps({"waiting": 0, "run": asked.get("run", 0)}))
        return SkipReason("no document is waiting to be asked for")
    if not queue.idle:
        # The cursor is left alone: the run already going is the one that
        # asked, and this tick has decided nothing.
        return SkipReason(
            f"{queue.outstanding} document(s) already queued or in progress"
        )

    run = asked.get("run", 0) + (waiting != asked.get("waiting"))
    context.update_cursor(json.dumps({"waiting": waiting, "run": run}))
    log.info("%d document(s) uploaded and never asked for; starting", waiting)
    return RunRequest(run_key=f"new-{run}-{waiting}")


def _statuses(backend_url: str, queues: tuple[str, ...]) -> dict[str, Any]:
    """Every stage's queue, asked for at once rather than one after another.

    Sequentially this is six GETs at the client's 15s timeout, and Dagster
    gives a sensor tick 60 before it kills it: a backend answering slowly
    would switch the watcher off exactly when the graph needs it.

    One Backend per thread, because a requests Session is a connection pool
    that is not documented thread-safe. A stage that cannot be reached is
    left out, so one unreachable route costs itself and not the tick.
    """
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        asked = {
            stage: pool.submit(Backend(backend_url).status, stage) for stage in queues
        }
    found = {}
    for stage, answer in asked.items():
        try:
            found[stage] = answer.result()
        except Exception as unreachable:  # noqa: BLE001 - one stage, not the tick
            log.warning("%s: no status, so nothing recorded: %s", stage, unreachable)
    return found


def _last(recorded: Any) -> tuple[int, int]:
    """What the cursor says a stage had worked and failed last tick.

    A bare int is the shape the cursor held before failures were watched,
    so a deployment upgrading into this reads its old cursor rather than
    re-reporting every stage from zero.
    """
    if isinstance(recorded, list):
        return recorded[0], recorded[1]
    return recorded or 0, 0


@sensor(
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.RUNNING,
    description="Records what each stage produced and what it failed, "
    "whoever set it going.",
)
def progress(context):
    """Reports what this orchestrator did not start, worked and failed alike.

    Without it Dagster sees only its own runs. Every other way of setting
    a stage going - `make extract`, POST /extraction/start, the Start
    button, a worker draining a queue somebody filled - moved rows and
    left the asset graph saying "never materialised", which is the one
    reading of that graph nobody should have to qualify.

    So the asset is no longer only a trigger. This watches the queues the
    same way the assets do, through /status, and files:

        an AssetMaterialization    when a stage's worked count has risen
        an AssetCheckEvaluation    when its failed count has moved at all

    The check is what makes the graph go red for work Dagster did not run.
    A materialisation alone cannot: a stage that failed every row produced
    nothing, so counting only what was produced reported silence for the
    one state worth interrupting somebody over. It is filed on a move in
    either direction, so a `retry` that clears the failures turns the
    asset green again rather than leaving it red until a run happens.

    The cursor holds the last worked and failed counts per stage, so a
    tick that finds neither moved reports nothing. First tick after an
    empty cursor records every stage that has produced or failed anything,
    which is how a graph that has been running for weeks without this
    catches up in one poll.

    RUNNING rather than STOPPED, unlike the schedule and `arrivals`:
    those two decide that work should happen, and a deployment should opt
    into that. This one only watches, and a watcher nobody switched on is
    a graph that is quietly wrong.
    """
    settings = Settings.load()
    seen = json.loads(context.cursor) if context.cursor else {}
    queues = _statuses(
        settings.backend_url, tuple(stage for _, stage, _ in stages.ALL_STAGES)
    )

    events, counted = [], {}
    for name, stage, unit in stages.ALL_STAGES:
        was_done, was_failed = _last(seen.get(stage))
        queue = queues.get(stage)
        if queue is None:
            counted[stage] = [was_done, was_failed]
            continue
        done, failed = stages.produced(queue.rows), queue.failed
        counted[stage] = [done, failed]

        # Nothing produced is nothing to materialise: a stage with an empty
        # queue has not materialised, it has not run.
        if done > was_done:
            events.append(
                AssetMaterialization(
                    asset_key=AssetKey(name),
                    description=f"{done} {unit}(s) worked",
                    metadata={
                        f"{unit}s done": done,
                        "since the last tick": done - was_done,
                        "failed": failed,
                        "queue": MetadataValue.json(queue.rows),
                    },
                )
            )
        if failed != was_failed:
            events.append(
                AssetCheckEvaluation(
                    asset_key=AssetKey(name),
                    check_name=stages.NOTHING_FAILED,
                    passed=failed == 0,
                    description=f"{failed} {unit}(s) left failed",
                    metadata={
                        "failed": failed,
                        "since the last tick": failed - was_failed,
                        "retry with": f"POST /{stage}/retry",
                        "queue": MetadataValue.json(queue.rows),
                    },
                )
            )

    context.update_cursor(json.dumps(counted))
    if not events:
        return SkipReason("no stage has produced or failed anything new")
    log.info("recorded %d event(s) nobody asked this to watch", len(events))
    return SensorResult(asset_events=events)


defs = Definitions(
    assets=[
        stages.parsed_documents,
        stages.passages,
        stages.facts,
        stages.topics,
        stages.questions,
        stages.assessments,
    ],
    asset_checks=[
        stages.parsed_documents_check,
        stages.passages_check,
        stages.facts_check,
        stages.topics_check,
        stages.questions_check,
        stages.assessments_check,
        stages.assessments_agreement,
    ],
    jobs=[corpus],
    schedules=[nightly],
    sensors=[arrivals, progress],
)
