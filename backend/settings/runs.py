"""What names one run, so two of them can be told apart.

`settings_version` names a CONFIGURATION. It is a digest of the stored
overrides, so two runs under one configuration carry the same value and are
indistinguishable in the data - which is exactly the position anybody
comparing runs is in. The A/B in `evaluation/README.md` changed a model and
a rule together, and the only reason the two runs could be separated
afterwards is that the first one's questions had been deleted first.

This is the other half. `settings_version` says what the run was configured
as; `run_id` says which run it was.

Three things can name it, and they are read in this order:

    RUN_ID              somebody named this run, and means it
    the row's trigger   whoever asked for the row carried an id, and the
                        worker adopts it for as long as it holds that row
    a uuid              nobody named it; one per process

The middle one is what joins a corpus to whatever set it going. A worker is
long-lived and drains whatever arrives, so an id per PROCESS - which is all
there used to be - named the container and not the work: a Dagster run, a
Start button and `make extract` all produced rows stamped with the same
worker, and nothing in the data said which of them had asked. `start` now
writes its own id onto the rows it queues and `StageQueue._claim` reads it
back, so the id on a produced row is the id of the thing that asked for it.

RUN_ID still wins, and has to: naming a run is a deliberate act for a
comparison, and the A/B in `evaluation/README.md` would otherwise be
renamed by whichever surface happened to press Start.

Named `runs` and not `run` because `run.py` is a command line in every
package here, this one included.
"""

from __future__ import annotations

import os
from functools import lru_cache
from uuid import uuid4

from telemetry import running, running_as

#: Set this to name a run rather than have it named. The A/B in
#: evaluation/README.md would have been
#: `make questions-rerun TOPIC=439 RUN_ID=b-gemma4-12b` against
#: `... RUN_ID=a-gpt-4.1`, and the comparison one command rather than two
#: tables read off a terminal and typed into a file by hand.
#:
#: Unset is the useful default, and its absence means something the way
#: EVAL_RUN_NAME's does: a uuid is unambiguous, and nothing has to be
#: decided in advance about whether a run will turn out to be worth
#: comparing.
VARIABLE = "RUN_ID"


@lru_cache(maxsize=1)
def _process_run() -> str:
    """Names this process, for the rows nobody asked for by name.

    Cached rather than threaded down through the factories, which is the
    difference between this and `settings_version`. A version can change
    while a process runs - that is what `reloading` in `stages/cli.py`
    watches for - so it has to be re-read and handed to whatever writes a
    row. A process cannot change: it IS the process.
    """
    return uuid4().hex


def triggered_by(trigger: str | None) -> None:
    """Records which run asked for the row this thread is about to work.

    Called by `StageQueue._claim` with whatever the row carries, including
    None: a row queued before this existed, or by a caller that named no
    run, must not inherit the trigger of the row worked before it.

    Held in `telemetry` rather than here, so one value answers both this and
    the `run.id` on every line the worker writes while it holds the row. Two
    context variables would be two answers to one question, and the lineage
    page's link into Grafana joins on exactly this.
    """
    running_as(trigger)


def run_id() -> str:
    """Names the run a row being written belongs to.

    RUN_ID where somebody set one, the trigger the claimed row carries
    where it has one, and this process otherwise. See the module docstring
    for why that is the order.
    """
    return named_run() or running() or _process_run()


def named_run() -> str | None:
    """The run's name where somebody chose one, and None where nobody did.

    The difference between a run that is going to be COMPARED and a run
    that is merely happening, which is what decides whether it gets a
    Phoenix project of its own. Every run still has an id - `run_id` above
    mints one - and that id is on every log line and every span whether or
    not anybody named it.

    A project per unnamed run was the original arrangement and it does not
    survive a worker: one uuid per process, one process per restart, and
    five hundred projects holding a handful of spans each.
    """
    return (os.environ.get(VARIABLE) or "").strip() or None
