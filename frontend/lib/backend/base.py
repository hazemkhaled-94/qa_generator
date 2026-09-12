"""Shared HTTP mechanics for every backend client."""

from __future__ import annotations

from typing import Any

import requests


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

    def _get(self, path: str, *, timeout: float = 10) -> requests.Response:
        """Sends a GET and raises on any error status.

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status.
        """
        response = self._session.get(f"{self._base_url}{path}", timeout=timeout)
        response.raise_for_status()
        return response

    def _delete(self, path: str, *, timeout: float = 60) -> requests.Response:
        """Sends a DELETE and raises on any error status.

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status.
        """
        response = self._session.delete(f"{self._base_url}{path}", timeout=timeout)
        response.raise_for_status()
        return response

    def _patch(
        self, path: str, *, timeout: float = 30, **kwargs: Any
    ) -> requests.Response:
        """Sends a PATCH and raises on any error status.

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status.
        """
        response = self._session.patch(
            f"{self._base_url}{path}", timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response

    def _post(
        self, path: str, *, timeout: float = 60, **kwargs: Any
    ) -> requests.Response:
        """Sends a POST and raises on any error status.

        Raises:
            requests.exceptions.RequestException: On a network failure or an
                error status, whose body carries the reason.
        """
        response = self._session.post(
            f"{self._base_url}{path}", timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response
