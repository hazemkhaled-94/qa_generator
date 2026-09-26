"""The corpus as one control, on the page a document lands on.

Every other control in this frontend belongs to one stage, and `lib/stage.py`
says so: a page carries the stage that produced what it lists, and no page
can reach another page's stage. This does not break that rule, because it
reaches no stage at all. It asks the orchestrator to decide that the pipeline
should run, which is the same decision `make corpus` and the nightly schedule
make, and the orchestrator is what then starts each stage in turn.

That distinction is the whole reason a Run button can exist here and a
"start chunking" button cannot. Running the corpus means waiting for one
stage to drain before starting the next, and nothing in a browser can hold
that wait: closing the tab would abandon it half way. So the wait lives in a
Dagster run, this asks for one, and what the panel then shows is that run
getting on with it.

**Without an orchestrator the panel still works.** Run is disabled and says
why; Start, Stop and Retry are queue verbs and go on doing exactly what the
per-stage buttons do, over every stage at once.
"""

from __future__ import annotations

import streamlit as st

from lib import page

#: How often the panel redraws. The same interval the stage controls use.
POLL_SECONDS = 3

#: Run, then the three that act on every queue, then the line of state.
_CONTROLS = [1.3, 1.2, 1.2, 1.2, 4.2]

#: What each Dagster run status means for somebody reading a page. Anything
#: not here is shown as Dagster spells it, lower-cased.
_RUN_READS = {
    "QUEUED": "queued",
    "NOT_STARTED": "starting",
    "STARTING": "starting",
    "STARTED": "running",
    "CANCELING": "stopping",
    "SUCCESS": "finished",
    "FAILURE": "failed",
    "CANCELED": "stopped",
}


def panel(client) -> None:
    """Draws the whole-pipeline controls, and goes on drawing them.

    Through `page.render`, because a fragment reruns on its own and so
    outside the guard the view was called under: an unreachable backend
    would otherwise replace the panel with a stack trace.
    """

    @st.fragment(run_every=POLL_SECONDS)
    def drawn() -> None:
        """Draws the controls, and keeps drawing them."""
        page.render(lambda: _controls(client))

    drawn()


def _controls(client) -> None:
    """Draws one pass of the panel."""
    state = client.state()
    if not state:
        st.caption("The backend is not answering.")
        return

    stages = state.get("stages") or []
    orchestration = state.get("orchestration") or {}
    available = bool(orchestration.get("available"))
    running = orchestration.get("running")
    working = bool(state.get("working"))
    failed = int(state.get("failed") or 0)
    ready = sum(
        (one.get("rows") or {}).get("new", 0)
        for one in stages
        if one.get("stage") != "assessment"
    )

    begun, started, halted, retried, reads = st.columns(
        _CONTROLS, vertical_alignment="center"
    )

    if begun.button(
        "Run the pipeline",
        key="pipeline-run",
        type="primary",
        width="stretch",
        disabled=not available or running is not None,
        help=_run_help(available, running),
    ):
        _report(client.run())

    # Every label here names the corpus, because the panel below this one
    # carries the same three verbs for the one stage this page owns. Two
    # controls reading `Start` on one page is a control nobody can use.
    if started.button(
        "Start every stage",
        key="pipeline-start",
        width="stretch",
        disabled=not ready,
        help=f"Queues the {ready:,} row(s) ready to be worked, in every "
        "stage. One stretch of the pipeline, not all of it."
        if ready
        else "Nothing is ready to be queued.",
    ):
        _report(client.act("start"))

    if halted.button(
        "Stop everything",
        key="pipeline-stop",
        width="stretch",
        disabled=not working and running is None,
        help="Takes back everything queued and stops the run behind it. "
        "Whatever a worker holds right now finishes."
        if working or running
        else "Nothing is queued or running.",
    ):
        _report(client.act("stop"))

    if retried.button(
        "Retry failures",
        key="pipeline-retry",
        width="stretch",
        disabled=not failed,
        help=f"Returns the {failed:,} failed row(s) of every stage to the queue."
        if failed
        else "Nothing has failed.",
    ):
        _report(client.act("retry"))

    reads.markdown(_reads(stages, running, working, failed))
    _automation(client, orchestration.get("automation") or {})


def _run_help(available: bool, running: dict | None) -> str:
    """Why Run is offered, or why it is not."""
    if not available:
        return (
            "This deployment has no orchestrator, so nothing can wait for "
            "one stage to drain before starting the next. Use Start, or "
            "`make corpus` from a terminal."
        )
    if running is not None:
        return (
            f"Run {running.get('id', '')[:8]} is already taking this corpus "
            "through. Stop it first, or wait for it."
        )
    return (
        "Takes every document through every stage, in order, waiting for "
        "each to finish. Only what has not been done already."
    )


def _reads(stages: list[dict], running: dict | None, working: bool, failed: int) -> str:
    """One line saying where the corpus has got to."""
    parts = []
    if running is not None:
        status = _RUN_READS.get(running.get("status", ""), "").strip()
        parts.append(f"run {running.get('id', '')[:8]} {status or 'running'}")
    elif working:
        parts.append("working")
    stalled = [one["stage"] for one in stages if (one.get("rows") or {}).get("failed")]
    if failed:
        parts.append(f"{failed:,} failed in {', '.join(stalled)}")
    if not parts:
        done = sum(
            count
            for one in stages
            for status, count in (one.get("rows") or {}).items()
            if status not in ("new", "pending", "in_progress", "failed")
        )
        parts.append(f"{done:,} row(s) done" if done else "nothing has run yet")
    return " · ".join(parts)


def _automation(client, automation: dict) -> None:
    """The two switches that start a run without being asked.

    Folded away, because turning one on is a decision made once and the
    four buttons above are the ones pressed daily. Off is what ships: a
    stack that starts working the moment it comes up is one nobody chose.
    """
    available = bool(automation.get("available"))
    on_arrival = bool(automation.get("on_arrival"))
    nightly = bool(automation.get("nightly"))
    running = [
        name for name, on in (("on upload", on_arrival), ("nightly", nightly)) if on
    ]

    # Outside the fold, because a fold is opened by somebody who expects
    # something in it. Nothing to switch is worth reading without the click.
    if not available:
        st.caption(
            "No orchestrator, so nothing can watch for uploads or run to a "
            "clock. `make corpus` from a terminal is the other way to take "
            "a corpus through in order."
        )
        return

    with st.expander(
        "Run it without being asked" + (f" · {', '.join(running)}" if running else "")
    ):
        chosen = st.toggle(
            "When a document is uploaded",
            value=on_arrival,
            key="pipeline-on-arrival",
            help="Starts a run whenever documents are sitting unasked-for. "
            "This is what makes the pipeline unattended.",
        )
        clocked = st.toggle(
            "Every night at 02:00 UTC",
            value=nightly,
            key="pipeline-nightly",
            help="A topic fit is corpus-wide and goes stale on every new "
            "document, which is what makes a clock worth having.",
        )
        if chosen != on_arrival or clocked != nightly:
            _report(client.automate(on_arrival=chosen, nightly=clocked))


def _report(outcome: tuple[bool, str]) -> None:
    """Shows what a verb did, and redraws the page under it."""
    done, said = outcome
    if done:
        st.toast(said)
        st.rerun()
    else:
        st.warning(said)
