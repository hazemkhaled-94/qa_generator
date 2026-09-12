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
    st.info("Coming soon.")


page.render(view)
