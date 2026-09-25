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


@pytest.fixture(autouse=True)
def nothing_remembered():
    """Empties Streamlit's data caches between runs.

    A `@st.cache_data` cache is per PROCESS, and a page that caches a fetch
    keys it on the arguments rather than on the client - which is the whole
    point of `_client` being underscored. Two tests driving one page with
    two scripted backends therefore share whatever the first of them
    cached, and the second reads an answer the client it was given never
    gave.

    Until now this was safe by accident: the one page that cached keyed on
    a document digest, and the fixtures happened to vary it.
    """
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def run_view(monkeypatch):
    """Runs one view with its backend clients stubbed."""
    from streamlit.testing.v1 import AppTest

    def run(name: str, timeout: float = 30, **clients):
        """Runs `frontend/views/{name}.py` and returns the finished app."""
        import lib.backend as module

        for factory, stub in clients.items():
            monkeypatch.setattr(module, factory, lambda stub=stub: stub)
        app = AppTest.from_file(str(VIEWS / f"{name}.py"), default_timeout=timeout)
        return app.run()

    return run


@pytest.fixture
def open_view(run_view):
    """Opens any view as the page object a test works it through."""
    from pages import OUTSIDE, Answers, answers, page_for, settings

    def opened(
        name: str,
        client: Answers | None = None,
        configured: Answers | None = None,
        **replaced,
    ):
        """Opens `frontend/views/{name}.py` with every client it reaches for.

        Three sorts of client, because a page holds up to three: the
        catalogue it lists from, the settings its configuration panel
        draws, and - on the two pages outside the catalogue - one of their
        own. All of them are given every time, because a panel that cannot
        reach the backend draws a caption instead of its controls and a
        page test should be looking at the controls.

        A keyword named for a client replaces that client; anything else
        replaces one of the catalogue's answers. Without the split a test
        handing this an unreachable `health_api` got a CATALOGUE scripted
        to answer `health_api`, and a health page with its real stub still
        attached.
        """
        named = [one for one in replaced if one.endswith("_api")]
        held = {
            factory: Answers(**scripted)
            for factory, scripted in OUTSIDE.get(name, {}).items()
        }
        held["catalog_api"] = client or Answers(**answers(**_without(replaced, named)))
        held["settings_api"] = configured or Answers(**settings())
        # Last, so naming a client outright beats both the default above and
        # the page's own stub - and so `catalog_api=` is one client rather
        # than the same keyword passed to run_view twice.
        held |= {factory: replaced[factory] for factory in named}
        return page_for(name)(run_view(name, **held), name)

    return opened


def _without(given: dict, names: list[str]) -> dict:
    """Whatever is left once the clients are taken out of the answers."""
    return {key: value for key, value in given.items() if key not in names}


@pytest.fixture
def page(request, open_view):
    """Runs the view this module is about, however the backend is scripted.

    The view is the module's own `VIEW`. Three test modules each carried a
    byte-identical copy of this closure differing only in that string.
    """
    view = getattr(request.module, "VIEW", None)
    assert view, f"{request.module.__name__} uses `page` without declaring VIEW"

    def run(**replaced):
        """Opens it with these answers replacing the defaults."""
        return open_view(view, **replaced)

    return run
