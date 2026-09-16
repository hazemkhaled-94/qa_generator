"""Shared layout for the view scripts.

Views are functions rather than classes: Streamlit re-runs the whole script
on every interaction, so nothing persists between calls except the HTTP
session, which lib.backend caches.

Every page is the same sequence of panels, top to bottom: the figures worth
seeing on arrival, an analysis fold nobody has to open, the one service this
page runs, search, filters, the table of everything, and the row a person
picked. A view that follows that order needs no layout of its own.

A panel is what separates one section from the next, and `panel` is the only
way to draw one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import streamlit as st

log = logging.getLogger(__name__)

#: What each queue status means, in one line. The same words appear on five
#: pages and on the system status panel.
STATUS_HELP = {
    "new": "Stored, never asked for.",
    "pending": "Queued, waiting for a worker.",
    "in_progress": "Held by a worker now.",
    "failed": "The worker could not finish it.",
    "parsed": "Converted into a structured document.",
    "chunked": "Split into passages.",
    "extracted": "Read for facts.",
    "modelled": "Fitted into the stored topic model.",
    "generated": "Questions have been written for it.",
}


def share(part: int, whole: int) -> str:
    """Renders a count as itself and its share of a whole, in one string."""
    if not whole:
        return f"{part:,}"
    return f"{part:,} ({part / whole:.0%})"


def header(title: str) -> None:
    """Renders the page's title."""
    st.html(f"<div class='qa-page-title'>{title}</div>")


@contextmanager
def panel(title: str) -> Iterator[Any]:
    """Opens one bordered section, labelled.

    The label is drawn inside the border rather than above it, so a section
    and its heading cannot drift apart.
    """
    box = st.container(border=True)
    with box:
        st.html(f"<div class='qa-panel-label'>{title}</div>")
        yield box


def stats(values: Mapping[str, tuple[str, str]]) -> None:
    """Renders one row of figures, each with the line that says what it is.

    Keyed by label, valued by `(figure, what it counts)`, so a caller cannot
    add a figure without saying what it is.
    """
    for column, (label, (value, explanation)) in zip(
        st.columns(len(values)), values.items(), strict=True
    ):
        column.metric(label, value, help=explanation, border=True)


def attributes(rows: Mapping[str, Any]) -> None:
    """Renders everything held about one item, as a two-column table."""
    st.dataframe(
        {"Field": list(rows), "Value": [written(one) for one in rows.values()]},
        width="stretch",
        hide_index=True,
        column_config={
            "Field": st.column_config.TextColumn(width="small"),
            "Value": st.column_config.TextColumn(width="large"),
        },
    )


def written(value: Any) -> str:
    """Renders one value the way a table cell should read it.

    Everything becomes a string: a column mixing a number with a dash is not
    a column a table can draw, and `None` should read as an em dash rather
    than as the word None.
    """
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(one) for one in value) or "—"
    return str(value)


def table(
    rows: list[dict],
    key: str,
    column_config: dict | None = None,
) -> int | None:
    """Draws the page's table and returns which row was picked, if any.

    Nothing below the table is drawn until a row is chosen, so this returns
    None on arrival and a position into `rows` afterwards.

    A position rather than an id: the selection Streamlit hands back is
    positional, and the caller already holds the rows it passed in.
    """
    picked = st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    chosen = picked["selection"]["rows"]
    # Guarded, not trusted: a selection outlives the rows it was made on, so
    # a filter that shortens the page leaves an index past the end of it.
    if not chosen or chosen[0] >= len(rows):
        return None
    return chosen[0]


def queue_rows(
    unit: str, counts: Mapping[str, int], states: tuple[str, ...]
) -> list[dict[str, str]]:
    """Builds the findings rows describing one stage's queue.

    The same five columns every other check in the fold carries, so one
    table can hold the queue and the quality checks together. Only `failed`
    has a verdict: the rest are where work is, not whether it went well.
    """
    total = sum(counts.values())
    return [
        {
            "Check": f"{unit} {state.replace('_', ' ')}",
            "Value": share(counts.get(state, 0), total),
            "Should be": "0" if state == "failed" else "—",
            "State": ("Attention" if counts.get(state) else "OK")
            if state == "failed"
            else "—",
            "What it means": STATUS_HELP[state],
        }
        for state in states
    ]


def findings(rows: list[dict[str, str]]) -> None:
    """Renders what a stage's own numbers say about themselves, as a table.

    Every check is listed, the ones that passed included: a check missing
    from a list is indistinguishable from a check nobody wrote.
    """
    if not rows:
        return
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Check": st.column_config.TextColumn(width="medium"),
            "What it means": st.column_config.TextColumn(width="large"),
        },
    )


def render(view: Callable[[], None]) -> None:
    """Runs one page, reporting a failure as a message not a traceback."""
    try:
        view()
    except Exception as exc:
        log.exception("page failed")
        st.error(f"Failed to load: {exc}")
