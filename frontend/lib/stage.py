"""The one service a page runs, and the controls that move its queue.

Each page carries exactly one stage, the one that produces what the page
lists: parsing on Documents, chunking on Passages, extraction on Facts,
topic modelling on Topics, generation on Questions. No page can reach
another page's stage.

Every stage offers the same two scopes. Without one the verb acts on
everything the stage owns; with one it acts on a single document, passage or
topic, and is drawn inside that item's own section.

Nothing starts by itself. A row this stage has never been asked to do sits
at `new`, and Start is what queues it; Stop puts back whatever has not
begun. The stage before this one cannot do either, and neither can this one
for the stage after.

The controls redraw on a timer, which is what the loading icon is for: a
verb only queues rows, the worker does the work, and the spinner runs until
that stage's queue is empty again.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from lib import page

#: How often the controls redraw while a worker is on a queue. Long enough
#: not to hammer the API, short enough that a long run looks alive.
POLL_SECONDS = 3

#: The four verbs, then the line saying where the queue stands.
_CONTROLS = [1.2, 1.2, 1.2, 1.2, 4.0]

#: The order the states are read out in, which is the order a row moves
#: through them.
_ORDER = ("new", "pending", "in_progress", "failed")


class Queue:
    """One stage as a page refers to it: its name and how it reads."""

    def __init__(self, name: str, label: str, unit: str, done: str) -> None:
        """Names a stage, what to call it, what it counts and its end state."""
        self.name = name
        self.label = label
        self.unit = unit
        self.done = done

    def describe(self, counts: dict[str, int]) -> str:
        """Reads this queue's counts back as one line."""
        parts = [
            f"{counts[state]:,} {state.replace('_', ' ')}"
            for state in (*_ORDER, self.done)
            if counts.get(state)
        ]
        return " · ".join(parts) or f"no {self.unit}"


def busy(counts: dict[str, int]) -> bool:
    """Reports whether this queue still has work in flight."""
    return bool(counts.get("pending", 0) or counts.get("in_progress", 0))


def service(client, queue: Queue, scope: tuple[str, str] | None = None) -> None:
    """Draws one stage's four verbs, each live only when it would act.

    `scope` is what to narrow to, as the pair the route takes: ("document",
    sha256), ("passage", id) or ("topic", id). Without one every verb acts
    on everything the stage owns, which is how a whole corpus is started at
    once.
    """
    _polling(lambda: _verbs(client, queue, scope))


def fit(client, queue: Queue) -> None:
    """Draws topic modelling's three verbs, which have no per-item form.

    Every topic is fitted jointly over one vocabulary, so there is no single
    topic to start, stop or refit.
    """
    _polling(lambda: _fit_verbs(client, queue))


def _polling(draw: Callable[[], None]) -> None:
    """Redraws one set of controls every few seconds, until the queue rests.

    Through page.render, because a fragment reruns on its own and so outside
    the guard the view was called under: an unreachable backend would replace
    the panel with a stack trace.
    """

    @st.fragment(run_every=POLL_SECONDS)
    def drawn() -> None:
        """Draws the controls, and goes on drawing them."""
        page.render(draw)

    drawn()


def _verbs(client, queue: Queue, scope: tuple[str, str] | None) -> None:
    """Draws the four queue verbs once, for whatever scope was given."""
    counts = client.stage_status(queue.name, scope)["rows"]
    total = sum(counts.values())
    waiting = counts.get("new", 0)
    queued = counts.get("pending", 0)
    failed = counts.get("failed", 0)
    everything = scope is None
    unit = queue.unit

    begun, halted, retried, redone, state = st.columns(
        _CONTROLS, vertical_alignment="center"
    )

    if begun.button(
        "Start all" if everything else "Start",
        key=_key("start", queue, scope),
        type="primary",
        disabled=not waiting,
        width="stretch",
        help=f"Queues the {waiting:,} {unit} never asked for."
        if waiting
        else f"No {unit} are waiting.",
    ):
        _act(client, queue.name, scope, "start")

    if halted.button(
        "Stop",
        key=_key("stop", queue, scope),
        disabled=not queued,
        width="stretch",
        help=f"Takes the {queued:,} queued {unit} back off the queue."
        if queued
        else f"No {unit} are queued.",
    ):
        _act(client, queue.name, scope, "stop")

    if retried.button(
        "Retry",
        key=_key("retry", queue, scope),
        disabled=not failed,
        width="stretch",
        help=f"Queues the {failed:,} failed {unit} again."
        if failed
        else f"No {unit} have failed.",
    ):
        _act(client, queue.name, scope, "retry")

    if redone.button(
        "Redo all" if everything else "Redo",
        key=_key("rerun", queue, scope),
        disabled=not total,
        width="stretch",
        help=f"Queues every {unit} again, finished ones included, and "
        "rebuilds what this stage made of them."
        if total
        else f"There are no {unit} to redo.",
    ):
        _act(client, queue.name, scope, "rerun")

    _state(state, queue, counts, failed)


def _fit_verbs(client, queue: Queue) -> None:
    """Draws the three verbs a fit takes."""
    counts = client.stage_status(queue.name)["rows"]
    queued = counts.get("pending", 0)
    failed = counts.get("failed", 0)
    running = busy(counts)

    begun, halted, retried, _, state = st.columns(
        _CONTROLS, vertical_alignment="center"
    )

    if begun.button(
        "Fit all",
        key=_key("discover", queue, None),
        type="primary",
        disabled=running,
        width="stretch",
        help="Queues one fit over every passage, one model per language. "
        "Replaces every stored topic."
        if not running
        else "A fit is already queued or running.",
    ):
        _act(client, queue.name, None, "discover")

    if halted.button(
        "Stop",
        key=_key("stop", queue, None),
        disabled=not queued,
        width="stretch",
        help=f"Takes the {queued:,} queued fit(s) back off the queue."
        if queued
        else "No fit is queued.",
    ):
        _act(client, queue.name, None, "stop")

    if retried.button(
        "Retry",
        key=_key("retry", queue, None),
        disabled=not failed,
        width="stretch",
        help="Queues a failed fit again." if failed else "No fit has failed.",
    ):
        _act(client, queue.name, None, "retry")

    _state(state, queue, counts, failed)


def _state(into, queue: Queue, counts: dict[str, int], failed: int) -> None:
    """Writes where the queue stands, with the loading icon while it moves."""
    icon = "<span class='qa-spin'></span>" if busy(counts) else ""
    tone = " qa-state-bad" if failed else ""
    into.html(f"<div class='qa-state{tone}'>{icon}{queue.describe(counts)}</div>")


def _key(action: str, queue: Queue, scope: tuple[str, str] | None) -> str:
    """Names a button uniquely, so two scopes of one verb are two widgets."""
    return "-".join((action, queue.name, *(scope or ())))


def _act(client, stage: str, scope: tuple[str, str] | None, action: str) -> None:
    """Sends one queue action and says what it did.

    The whole app, not just this fragment: the figures at the top of the
    page count the same queue these verbs just moved.
    """
    answer = client.stage_action(stage, action, scope)
    st.toast(answer.get("detail") or f"{stage}: {action}")
    st.rerun(scope="app")


def removal(
    key: str,
    note: str,
    buttons: dict[str, tuple[str, str, str]],
    subject: str = "",
) -> str | None:
    """Draws the delete box, with its buttons inside it.

    `buttons` maps a name to its label, its tooltip and the sentence the
    confirmation asks. The first click only arms the choice; a second,
    separately labelled one is what this returns, because none of it can be
    undone.

    `subject` is what is being deleted, so arming a deletion on one item and
    then picking another disarms it.
    """
    slot = f"{key}-armed"
    armed = st.session_state.get(slot)
    chosen = None

    with page.panel("Delete"):
        st.html(f"<div class='qa-danger-note'>{note}</div>")
        columns = st.columns(max(len(buttons), 3))
        for column, (name, (label, explanation, _)) in zip(
            columns, buttons.items(), strict=False
        ):
            if column.button(
                label, key=f"danger-{key}-{name}", width="stretch", help=explanation
            ):
                # Remembered and acted on in this same run: the confirmation
                # is drawn below, so arming needs no rerun to become visible.
                st.session_state[slot] = armed = (name, subject)

        if armed and armed[1] == subject and armed[0] in buttons:
            st.warning(buttons[armed[0]][2])
            confirm, cancel, *_ = st.columns(max(len(buttons), 3))
            if confirm.button(
                "Yes, delete", key=f"danger-confirm-{key}", width="stretch"
            ):
                del st.session_state[slot]
                chosen = armed[0]
            if cancel.button("Cancel", key=f"cancel-{key}", width="stretch"):
                del st.session_state[slot]
                st.rerun()
    return chosen
