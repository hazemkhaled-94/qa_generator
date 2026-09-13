"""Backend calls made by the System health page."""

from __future__ import annotations

import logging

import requests

from lib.backend.base import Endpoint

log = logging.getLogger(__name__)


class HealthApi(Endpoint):
    """What the System health page asks the backend to report."""

    def reachable(self) -> tuple[bool, str]:
        """Checks whether the backend answers at all.

        Separate from `components`, so the page can tell the backend being
        down from the backend's database being down.
        """
        try:
            self._request("GET", "/health", timeout=5)
        except requests.exceptions.RequestException as exc:
            log.warning("backend unreachable: %s", exc)
            return False, str(exc)
        return True, "Reachable."

    def components(self) -> dict[str, dict]:
        """Fetches the state of everything behind the API."""
        try:
            return self._request("GET", "/status").json()
        except requests.exceptions.RequestException as exc:
            log.warning("cannot read backend status: %s", exc)
            return {}
