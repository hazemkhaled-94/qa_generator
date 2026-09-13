"""Shared helpers for the Streamlit frontend.

Telemetry is configured here, on import, because this is the one thing every
view reaches: Streamlit runs each page as a script of its own, so there is no
entry point below `app.py` that all of them pass through, and `app.py` is
re-executed on every interaction while a module import is not.
"""

import telemetry

telemetry.configure("frontend")
