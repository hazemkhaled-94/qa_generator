"""Shared HTTP mechanics for every backend client."""

from __future__ import annotations

from typing import Any

import requests

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

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status, whose body carries the reason.
        """
        response = self._session.request(
            method,
            f"{self._base_url}{path}",
            timeout=_TIMEOUTS[method] if timeout is None else timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response
