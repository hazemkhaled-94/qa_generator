"""The queue strip a stage's own page shows.

Each stage's controls sit on the page showing what that stage produces -
parsing on Documents, chunking on Passages, extraction on Facts, topics on
Topics - rather than on a page of their own.

One shape everywhere: the stage's status on the left, its controls on the
right, its progress underneath. Topics passes a different set of controls
into the same strip, because what it can be asked to do differs.

Nothing starts by itself. A row this stage has never been asked to do sits
at `new`, and Start is what queues it; Stop puts back whatever has not
begun. The stage before this one cannot do either, and neither can this one
for the stage after.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from lib import backend, catalog

#: How often the strip redraws while a worker is on the queue. Long enough
#: not to hammer the API, short enough that a long run looks alive.
POLL_SECONDS = 3

#: Width of the status pill against the four controls beside it.
CONTROLS = [3.2, 1.2, 1.2, 1.0, 1.0]


def status(state: dict) -> str:
    """Renders the pill saying whether a worker is on this queue."""
    return catalog.stage_pill("working" if state["working"] else "idle")


def running(counts: dict) -> bool:
    """Says whether this stage has work queued or in hand."""
    return bool(counts.get("pending", 0) or counts.get("in_progress", 0))


def panel(
    stage: str,
    unit: str,
    done_status: str,
    *,
    start: Callable[[object], dict] | None = None,
    rerun: bool = True,
) -> None:
    """Renders one stage's queue, its progress and its controls.

    A fragment that reruns on its own, so progress moves while the worker
    works rather than the page sitting still until it finishes.
    """

    @st.fragment(run_every=POLL_SECONDS)
    def _draw() -> None:
        """Draws the strip, and keeps drawing it."""
        client = backend.catalog_api()
        state = client.stage_status(stage)
        counts = state["rows"]
        total = sum(counts.values())
        waiting = counts.get("new", 0)
        failed = counts.get("failed", 0)
        live = running(counts)
        # Topic modelling has no `new`: a fit is asked for, which creates the
        # row already queued, so Start is offered whenever none is waiting.
        startable = waiting if start is None else (0 if live else 1)

        left, retry, redo, stop, begin = st.columns(
            CONTROLS, vertical_alignment="center"
        )
        left.markdown(status(state), unsafe_allow_html=True)
        if retry.button(
            "Retry failed",
            key=f"retry-{stage}",
            disabled=not failed,
            width="stretch",
            help=f"Queue the {failed:,} failed {unit} again."
            if failed
            else "Nothing has failed.",
        ):
            _act(client, stage, "retry")
        if rerun and redo.button(
            f"Redo all {unit}",
            key=f"rerun-{stage}",
            disabled=not total,
            width="stretch",
            help="Queue every one again, finished included. For when the code "
            "behind this stage has changed."
            if total
            else f"There are no {unit} to redo.",
        ):
            _act(client, stage, "rerun")
        if stop.button(
            "Stop",
            key=f"stop-{stage}",
            disabled=not live,
            width="stretch",
            help="Take back what has not begun. The one in hand finishes."
            if live
            else "Nothing is running.",
        ):
            _act(client, stage, "stop")
        if begin.button(
            "Start",
            key=f"start-{stage}",
            type="primary",
            disabled=not startable,
            width="stretch",
            help=f"Queue the {waiting:,} {unit} not done yet."
            if start is None and waiting
            else "Queue a fit over the whole corpus. Replaces every existing topic."
            if start is not None and startable
            else f"No {unit} are waiting. Redo or retry to queue finished ones."
            if start is None
            else "A fit is already queued.",
        ):
            if start is None:
                _act(client, stage, "start")
            else:
                st.toast(start(client).get("detail", "queued"))
                st.rerun()

        if not total:
            return

        done = counts.get(done_status, 0)
        label = f"{done:,} of {total:,} {unit} {done_status}"
        for count, word in (
            (counts.get("in_progress", 0), "in progress"),
            (counts.get("pending", 0), "queued"),
            (waiting, "not started"),
            (failed, "failed"),
        ):
            if count:
                label += f" · {count:,} {word}"
        st.progress(done / total, text=label)

    _draw()


def _act(client, stage: str, action: str) -> None:
    """Sends one queue action and says what it did."""
    answer = client.stage_action(stage, action)
    st.toast(answer.get("detail") or f"{stage}: {action}")
    st.rerun()
