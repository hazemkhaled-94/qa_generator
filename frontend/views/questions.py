"""Questions view."""

from __future__ import annotations

import streamlit as st

from lib import page


def view() -> None:
    """Renders the questions page."""
    page.header(
        "Questions",
        "The questions and answers generated from the verified facts.",
    )
    page.section(
        "Not built yet",
        "This page will follow the same shape as the others: figures at the "
        "top, then the listing, then the one question you pick and the "
        "controls that act on it alone.",
    )
    st.info(
        "Coming soon. Only facts that passed every check are usable here, so "
        "the Facts page is where the input to this is judged."
    )


page.render(view)
