"""Shared behaviour for the view scripts.

Views are functions rather than classes: Streamlit re-runs the whole script
on every interaction, so nothing persists between calls except the HTTP
session, which lib.backend caches.

Every page is laid out the same way, top to bottom: `header`, then the
stage's queue strip where there is one, then `metrics`, then the toolbar
`lib.catalog.filters` draws, then the table, then anything about a single
row. A view that follows that order needs no layout of its own.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from lib import config

log = config.get_logger("page")


def header(title: str, description: str | None = None) -> None:
    """Renders the page's title block."""
    st.html(
        f"<div class='qa-page-title'>{title}</div>"
        + (f"<div class='qa-page-sub'>{description}</div>" if description else "")
    )


def metrics(values: dict[str, str]) -> None:
    """Renders one row of figures, in the same place on every page."""
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values.items(), strict=True):
        column.metric(label, value)


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
