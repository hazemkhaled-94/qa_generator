"""The queue figures and the controls that act on one item at a time.

Each stage appears on the page showing what it produces - parsing and
chunking on Documents, extraction on Documents, Passages and Facts, topic
modelling on Topics.

Two shapes, and the split between them is deliberate. `overview` is a
fragment that redraws itself every few seconds: it carries figures and no
buttons, so redrawing it can interrupt nothing. `controls` carries the
buttons and does not poll, because a strip that redraws every three seconds
took a confirmation away mid-decision.

Nothing starts by itself. A row this stage has never been asked to do sits
at `new`, and Start is what queues it; Stop puts back whatever has not
begun. The stage before this one cannot do either, and neither can this one
for the stage after.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from lib import page

#: How often the figures redraw while a worker is on a queue. Long enough not
#: to hammer the API, short enough that a long run looks alive.
POLL_SECONDS = 3

#: The label, then the four verbs, then the state they left the item in.
CONTROLS = [2.3, 1.15, 1.15, 1.15, 1.15, 2.6]

#: The delete buttons and the pair that confirms one. Wider than a verb
#: column: these say what they delete rather than reading `Delete`, and a
#: truncated label on the one irreversible control is the worst place for it.
DANGER = [2.6, 2.1, 4.8]

#: The two controls that decide one row's fate, and the space after them.
#: Narrower than DANGER: neither deletes anything.
VERDICT = [1.3, 1.3, 5.4]

#: The order the states are read out in, which is the order a row moves
#: through them.
_ORDER = ("new", "pending", "in_progress", "failed")

#: What each verb is for, as the prefix its button is keyed with. The
#: stylesheet colours on that prefix, so a control says what it does by being
#: keyed for it and no call site has to remember to pass a style.
#:
#: Each is graded by what it costs if pressed by mistake. Most are queue
#: verbs; `accept` and `reject` decide one stored row's fate and are graded
#: the same way, because a person reads the colour and not the list:
#:
#:   go       starts work that was going to be done anyway
#:   recover  re-queues only what already failed
#:   halt     withdraws from the queue; nothing already finished is lost
#:   redo     throws finished output away and rebuilds it
#:   danger   deletes data, and cannot be undone
INTENT = {
    "start": "go",
    "discover": "go",
    "accept": "go",
    "retry": "recover",
    "stop": "halt",
    "reject": "halt",
    "rerun": "redo",
}


#: Read out in the help of every section that draws controls. The colours are
#: graded by consequence, and a grading nobody can decode is decoration.
COLOUR_KEY = (
    "The controls are coloured by what pressing one costs. Green starts work "
    "that was going to be done anyway. Blue re-queues only what already "
    "failed. Grey withdraws from the queue, losing nothing already finished. "
    "Amber throws finished output away and rebuilds it. Red deletes data and "
    "cannot be undone. A control that would do nothing right now is greyed "
    "out rather than hidden, so the row keeps its shape."
)


def key_for(action: str, *parts: str) -> str:
    """Names a button so the stylesheet can colour it by what it does."""
    return "-".join((INTENT[action], action, *parts))


class Queue:
    """One stage as a page refers to it: its name and how it reads."""

    def __init__(self, name: str, label: str, unit: str, done: str) -> None:
        """Names a stage, what to call it, what it counts and its end state."""
        self.name = name
        self.label = label
        self.unit = unit
        self.done = done

    def describe(self, counts: dict[str, int]) -> str:
        """Reads this queue's counts back as a sentence."""
        parts = [
            f"{counts[state]:,} {state.replace('_', ' ')}"
            for state in (*_ORDER, self.done)
            if counts.get(state)
        ]
        return " · ".join(parts) or f"no {self.unit}"


def overview(
    client,
    queues: list[Queue],
    figures: Callable[[dict[str, dict[str, int]]], dict[str, tuple]],
    *,
    scope: tuple[str, str] | None = None,
) -> None:
    """Draws the page's figures and one bar per stage, and keeps drawing them.

    A fragment that reruns on its own, so a long run moves on the page rather
    than sitting still until somebody clicks something. Read-only: every
    button is in `controls`, which does not poll.

    `figures` turns the counts into the row of statistics the page wants,
    because what is worth showing differs per page while the polling does
    not.
    """

    def _figures() -> None:
        """Draws the figures once."""
        counts = {
            queue.name: client.stage_status(queue.name, scope)["rows"]
            for queue in queues
        }
        page.metrics(figures(counts))
        for queue in queues:
            rows = counts[queue.name]
            total = sum(rows.values())
            if total:
                st.progress(
                    rows.get(queue.done, 0) / total,
                    text=f"{queue.label}: {queue.describe(rows)}",
                )

    # Through page.render, because a fragment reruns on its own and so
    # outside the guard the view was called under: an unreachable backend
    # replaced this panel with a stack trace.
    @st.fragment(run_every=POLL_SECONDS)
    def _draw() -> None:
        """Draws the figures, and goes on drawing them."""
        page.render(_figures)

    _draw()


def controls(
    client,
    queue: Queue,
    scope: tuple[str, str],
    *,
    redo: bool = True,
) -> None:
    """Draws the four verbs for one item, each enabled only when it would act.

    `scope` is what the item is, as the pair the route takes: ("document",
    sha256) or ("passage", id). Nothing here is corpus-wide - the same verb
    over everything is `corpus_controls`.
    """
    kind, value = scope
    unit = queue.unit
    counts = client.stage_status(queue.name, scope)["rows"]
    total = sum(counts.values())
    waiting = counts.get("new", 0)
    queued = counts.get("pending", 0)
    failed = counts.get("failed", 0)
    held = counts.get("in_progress", 0)

    def key(action: str) -> str:
        """Keys one of this item's buttons for its verb and its colour."""
        return key_for(action, queue.name, kind, str(value))

    labelled, begun, redone, halted, retried, state = st.columns(
        CONTROLS, vertical_alignment="center"
    )
    labelled.html(f"<div class='qa-control-label'>{queue.label}</div>")

    if begun.button(
        "Start",
        key=key("start"),
        disabled=not waiting,
        width="stretch",
        help=f"Queue the {waiting:,} {unit} of this {kind} that {queue.label} "
        "has never been asked to do. A worker picks them up on its next poll."
        if waiting
        else f"Nothing here is waiting: every {unit} of this {kind} has "
        "already been asked for. Use Redo to run them again.",
    ):
        _act(client, queue.name, scope, "start")

    if redo and redone.button(
        "Redo",
        key=key("rerun"),
        disabled=not total,
        width="stretch",
        help=f"Queue every {unit} of this {kind} again, finished ones "
        f"included, and rebuild what {queue.label} produced from them. For "
        "when the code or the model behind the stage has changed. Whatever a "
        "worker holds right now is skipped."
        if total
        else f"This {kind} has no {unit} to redo.",
    ):
        _act(client, queue.name, scope, "rerun")

    if halted.button(
        "Stop",
        key=key("stop"),
        disabled=not queued,
        width="stretch",
        help=f"Take the {queued:,} queued {unit} back off the queue, to `new`."
        + (f" The {held:,} a worker holds now still finishes." if held else "")
        if queued
        else f"Nothing of this {kind} is queued.",
    ):
        _act(client, queue.name, scope, "stop")

    if retried.button(
        "Retry",
        key=key("retry"),
        disabled=not failed,
        width="stretch",
        help=f"Queue the {failed:,} failed {unit} again, clearing the error "
        "recorded against each."
        if failed
        else f"Nothing of this {kind} has failed.",
    ):
        _act(client, queue.name, scope, "retry")

    # Coloured rather than pilled: the counts already name the state, so a
    # pill beside them would read "failed · 3 failed". What a pill was there
    # for is making a failure visible without reading, which the colour does.
    tone = " qa-state-bad" if failed else ""
    state.html(f"<div class='qa-control-label{tone}'>{queue.describe(counts)}</div>")


def corpus_controls(
    client, stage: str, verbs: dict[str, tuple[str, bool, str]]
) -> None:
    """Draws verbs that act on the whole corpus rather than on one item.

    Only topic modelling uses this. The factorisation fits every topic
    jointly over one vocabulary, so there is no single topic to start, stop or
    refit: the stage has no scope to narrow to, and the page says so beside
    these. The keys are the route's own verbs, which is why the first is
    `discover` rather than `start`; it is coloured as a start either way,
    because that is what it does.
    """
    columns = st.columns([1.3] * len(verbs) + [CONTROLS[-1]])
    for index, (action, (label, enabled, explanation)) in enumerate(verbs.items()):
        if columns[index].button(
            label,
            key=key_for(action, "corpus", stage),
            disabled=not enabled,
            width="stretch",
            help=explanation,
        ):
            _act(client, stage, None, action)


def _act(client, stage: str, scope: tuple[str, str] | None, action: str) -> None:
    """Sends one queue action and says what it did."""
    answer = client.stage_action(stage, action, scope)
    st.toast(answer.get("detail") or f"{stage}: {action}")
    st.rerun()
