"""Backend calls made by the pipeline panel.

The corpus as one thing: every stage's queue in one read, the three verbs
that act on all of them, and the run that takes the whole thing through.

Its own client rather than four more methods on `CatalogApi`: that one is
what the Documents, Passages and Facts pages use to read what the pipeline
produced, and this is what asks for the pipeline to run. A page holding both
is holding two different things.
"""

from __future__ import annotations

import logging

import requests

from lib.backend.base import Endpoint

log = logging.getLogger(__name__)


class PipelineApi(Endpoint):
    """What the pipeline panel asks the backend to do."""

    def state(self) -> dict:
        """Every stage's queue, the run in flight, and what is automated.

        Empty when the backend cannot be reached, which the panel reports
        for itself: this is polled every few seconds and an exception here
        would replace the page with a stack trace.
        """
        try:
            return self._request("GET", "/pipeline").json()
        except requests.exceptions.RequestException as exc:
            log.warning("cannot read the pipeline state: %s", exc)
            return {}

    def run(self) -> tuple[bool, str]:
        """Asks for the whole corpus to be taken through, in order.

        Returns:
            Whether a run started, and what to tell the person either way.
            A refusal is answered rather than raised because every one of
            them is something they can act on: a run already going, or a
            deployment with no orchestrator.
        """
        try:
            started = self._request("POST", "/pipeline/run").json()
        except requests.exceptions.RequestException as exc:
            return False, _reason(exc, "The pipeline could not be started.")
        return True, f"Run {started['id'][:8]} started."

    def act(self, action: str) -> tuple[bool, str]:
        """Runs one verb over every stage at once.

        `start` queues everything ready to be worked, `stop` takes back
        everything queued and stops any run behind it, `retry` returns every
        stage's failures.
        """
        try:
            done = self._request("POST", f"/pipeline/{action}").json()
        except requests.exceptions.RequestException as exc:
            return False, _reason(exc, f"Could not {action} the pipeline.")
        return True, done["detail"]

    def automate(self, **switches: bool) -> tuple[bool, str]:
        """Switches unattended running on or off.

        `on_arrival` starts a run when documents are waiting; `nightly` runs
        the corpus at 02:00 UTC. Anything left out is left alone.
        """
        try:
            self._request("PUT", "/pipeline/automation", json=switches)
        except requests.exceptions.RequestException as exc:
            return False, _reason(exc, "Could not change what runs by itself.")
        return True, "Saved."


def _reason(exc: requests.exceptions.RequestException, fallback: str) -> str:
    """The backend's own wording for a refusal, or a fallback.

    Every deliberate refusal carries `{"code", "detail"}`; a network failure
    carries no body at all, and neither does a 500.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return f"{fallback} The backend is not answering."
    try:
        return response.json().get("detail") or fallback
    except ValueError:
        return fallback
