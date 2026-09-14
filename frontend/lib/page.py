"""Shared behaviour for the view scripts.

Views are functions rather than classes: Streamlit re-runs the whole script
on every interaction, so nothing persists between calls except the HTTP
session, which lib.backend caches.

Every page is laid out the same way, top to bottom: `header`, then the
statistics for whatever the page is about, then the toolbar
`lib.catalog.filters` draws, then the table, then the one row a person
picked and the controls that act on it. A view that follows that order needs
no layout of its own.

Nothing here renders a label without an explanation beside it. `metrics`,
`section` and `findings` all take the help text as part of the value, so a
figure cannot reach the page without saying what it counts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import streamlit as st

log = logging.getLogger(__name__)

#: What each queue status means, in one place: the same words appear on four
#: pages and on the system status panel.
STATUS_HELP = {
    "new": "Stored but never asked for. No worker looks at a row in this "
    "state; Start is what makes it claimable.",
    "pending": "Queued and waiting. The next free worker will claim it.",
    "in_progress": "Claimed by a worker right now.",
    "failed": "The worker could not finish it; the reason is recorded on the "
    "row. Retry puts it back on the queue.",
    "parsed": "Converted from the uploaded file into a structured document.",
    "chunked": "Split into passages, each with its sentences and lemmas read.",
    "extracted": "Read for facts. Facts that failed a check are stored too.",
    "modelled": "Fitted into the topic model currently stored.",
    "generated": "Questions have been written for this topic. Questions a "
    "gate rejected are stored too.",
}


def share(part: int, whole: int) -> str:
    """Renders a count as itself and its share of a whole, in one string.

    For running text and table cells. Every page states a proportion this
    way, so 0 of 0 reads as 0 rather than as a division by zero or as 100%.
    """
    if not whole:
        return f"{part:,}"
    return f"{part:,} ({part / whole:.0%})"


def portion(part: int, whole: int) -> tuple[str, str]:
    """Splits a proportion into the figure and the line beneath it.

    For `metrics`, which has room for a four-digit count or a percentage but
    not both: `2,469 (75%)` was truncated to `2,469 (7…` at five figures to a
    row. Splat it into the value: `(*portion(a, b), "what it counts")`.
    """
    if not whole:
        return f"{part:,}", "of nothing yet"
    return f"{part:,}", f"{part / whole:.0%} of {whole:,}"


def header(title: str, description: str | None = None) -> None:
    """Renders the page's title block."""
    st.html(
        f"<div class='qa-page-title'>{title}</div>"
        + (f"<div class='qa-page-sub'>{description}</div>" if description else "")
    )


def section(title: str, help: str) -> None:
    """Renders a section label with the tooltip explaining what it covers.

    `st.markdown` rather than `st.subheader`: it is the heading call that
    carries `help`, and the stylesheet gives the h5 it emits the section
    look.
    """
    st.markdown(f"##### {title}", help=help)


def metrics(values: dict[str, tuple]) -> None:
    """Renders one row of figures, each with the tooltip that explains it.

    Keyed by label, valued by `(figure, explanation)` or by `(figure, line
    beneath it, explanation)`, so a caller cannot add a figure without saying
    what it is. The explanation is always last.

    The second line is drawn as a delta with its colouring off: it is a
    second reading of the same figure, not a change in it, so an arrow and a
    green would both be lies.
    """
    columns = st.columns(len(values))
    for column, (label, spec) in zip(columns, values.items(), strict=True):
        value, beneath, explanation = (
            spec if len(spec) == 3 else (spec[0], None, spec[1])
        )
        column.metric(
            label,
            value,
            delta=beneath,
            delta_color="off",
            help=explanation,
            border=True,
        )


def findings(title: str, help: str, rows: list[dict[str, str]]) -> None:
    """Renders what a stage's own numbers say about themselves, as a table.

    The place where a warning used to be a coloured box with one sentence in
    it. A reader needs the measurement, what it should be and what it means
    to judge whether anything is wrong, so all three are columns and every
    check is listed - the ones that passed included, since a check missing
    from a list is indistinguishable from a check nobody wrote.
    """
    if not rows:
        return
    section(title, help)
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
    """Runs one page, reporting a failure as a message not a traceback.

    Every view module ends with a call to this. Streamlit runs each view as
    a script, so an escaping exception would replace the page with a stack
    trace.
    """
    try:
        view()
    except Exception as exc:
        log.exception("page failed")
        st.error(f"Failed to load: {exc}")
