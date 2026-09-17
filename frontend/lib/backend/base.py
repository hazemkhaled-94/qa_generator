"""Shared HTTP mechanics for every backend client."""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

#: How long each verb waits, by what it costs the backend. A read is a query;
#: a delete walks three stores; a post carries a file.
_TIMEOUTS = {"GET": 10.0, "PATCH": 30.0, "DELETE": 60.0, "POST": 60.0}


class Endpoint:
    """One group of backend endpoints, serving one page.

    Holds the base URL, the session and the request mechanics; a subclass
    adds the calls its page makes. No client learns what runs behind the
    URL.
    """

    def __init__(self, base_url: str, session: requests.Session) -> None:
        """Initialises the client."""
        self._base_url = base_url
        self._session = session

    def _request(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> requests.Response:
        """Sends one request and raises on any error status.

        A failure is logged before it is raised. The page above renders the
        exception as a message in the browser and nothing else, so without
        this the only record of a backend the frontend could not reach was
        on somebody's screen.

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status, whose body carries the reason.
        """
        about = {"http.request.method": method, "url.path": path}
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                timeout=_TIMEOUTS[method] if timeout is None else timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            status = getattr(exc.response, "status_code", None)
            log.warning(
                "%s %s failed: %s",
                method,
                path,
                exc,
                extra=about | ({"http.response.status_code": status} if status else {}),
            )
            raise
        return response
