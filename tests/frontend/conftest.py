"""The frontend under Streamlit's own test runner.

Each view is a script Streamlit runs top to bottom on every interaction, so
an `AppTest` runs it the same way. The backend is stubbed: what these cover
is what the page does with an answer, including the answers that are
refusals, which is the half no integration test reaches.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ROOT / "frontend/views"

#: What frontend/lib/config.py reads at import. Set before a view is run,
#: because reading them is what importing it does.
os.environ.setdefault("BACKEND_URL", "http://backend.invalid")
os.environ.setdefault("PAGE_SIZE", "25")


class Answers:
    """A backend client that answers from a script, and records the asks."""

    def __init__(self, **answers: object) -> None:
        """Initialises the client with one answer per method name."""
        self._answers = answers
        self.asked: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        """Answers whatever the page calls, or raises what it was given."""
        if name not in self._answers:
            raise AttributeError(name)

        def answer(*args, **kwargs):
            """Records the call and gives back the scripted answer."""
            self.asked.append((name, args, kwargs))
            prepared = self._answers[name]
            if isinstance(prepared, Exception):
                raise prepared
            return prepared(*args, **kwargs) if callable(prepared) else prepared

        return answer


@pytest.fixture
def run_view(monkeypatch):
    """Runs one view with its backend clients stubbed."""
    from streamlit.testing.v1 import AppTest

    def run(name: str, timeout: float = 30, **clients: Answers):
        """Runs `frontend/views/{name}.py` and returns the finished app."""
        import lib.backend as module

        for factory, stub in clients.items():
            monkeypatch.setattr(module, factory, lambda stub=stub: stub)
        app = AppTest.from_file(str(VIEWS / f"{name}.py"), default_timeout=timeout)
        return app.run()

    return run


@pytest.fixture
def open_view(run_view):
    """Opens one view as the page object a test works it through."""
    from pages import View, answers, settings

    def opened(
        name: str,
        client: Answers | None = None,
        configured: Answers | None = None,
        **replaced,
    ) -> View:
        """Opens `frontend/views/{name}.py` with the backend stubbed.

        Two clients, because every page draws a configuration panel beside
        the service it runs and that panel holds a client of its own.
        """
        return View(
            run_view(
                name,
                catalog_api=client or Answers(**answers(**replaced)),
                settings_api=configured or Answers(**settings()),
            ),
            name,
        )

    return opened
