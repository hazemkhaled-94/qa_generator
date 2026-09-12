"""System health view."""

from __future__ import annotations

from html import escape
from itertools import batched

import streamlit as st

from lib import backend, config, page

#: Cards per row. Fixed, so a card is the same width however many components
#: the backend reports and a new one wraps instead of squeezing the rest.
_PER_ROW = 4


def card(column, label: str, ok: bool, detail: str) -> None:
    """Renders one status card."""
    pill = "pill-ok" if ok else "pill-no"
    column.html(
        "<div class='qa-card'>"
        f"<div class='qa-card-label'>{escape(label)}</div>"
        f"<div class='qa-card-pill'><span class='pill {pill}'>"
        f"{'ok' if ok else 'failed'}</span></div>"
        f"<div class='qa-card-detail'>{escape(detail)}</div>"
        "</div>"
    )


def view() -> None:
    """Renders the system health page.

    Draws a card per component the backend reports, then its numbers, so a
    new component appears without a change here.
    """
    page.header(
        "System health",
        "Whether every component behind the API is reachable, and what each "
        "one currently holds.",
    )

    client = backend.health_api()
    api_ok, api_message = client.reachable()
    components = client.components()

    cards = [("Backend", api_ok, f"{config.BACKEND_URL} — {api_message}")]
    cards += [
        (name.replace("_", " ").title(), state["ok"], state["detail"])
        for name, state in components.items()
    ]

    for row in batched(cards, _PER_ROW):
        # strict=False: the last row is short, and its cards keep the width
        # the full rows have rather than spreading to fill it.
        for column, (label, ok, detail) in zip(st.columns(_PER_ROW), row, strict=False):
            card(column, label, ok, detail)

    if not components:
        st.warning("The backend is unreachable, so there is nothing to report.")
        return

    for name, state in components.items():
        if not state["metrics"]:
            continue
        st.divider()
        st.html(f"<div class='qa-section'>{escape(name.replace('_', ' '))}</div>")
        page.metrics(
            {
                label.replace("_", " ").title(): "—" if value is None else f"{value:,}"
                for label, value in state["metrics"].items()
            }
        )


page.render(view)
