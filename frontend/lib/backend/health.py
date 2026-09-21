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

    def services(self) -> list[dict]:
        """Fetches every container the deployment runs, and where to open it.

        A second call rather than a field on `components`: that reports what
        the pipeline holds and this reports what is up, and a page showing
        both has to say which is which. Empty when the backend cannot be
        reached, which the page already reports on its own.
        """
        try:
            return self._request("GET", "/services", timeout=20).json()
        except requests.exceptions.RequestException as exc:
            log.warning("cannot read the service list: %s", exc)
            return []
