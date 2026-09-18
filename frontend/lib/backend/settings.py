"""What each service is configured to do, read and changed.

Its own client: six pages draw a configuration panel and two of them hold
no catalogue client at all.
"""

from __future__ import annotations

from lib.backend.base import Endpoint


class SettingsApi(Endpoint):
    """The settings routes, one service at a time."""

    def settings(self, service: str) -> dict:
        """Fetches what one service is configured to do.

        Everything a control needs comes with it: the type, the bounds, and
        the closed set of values where there is one.
        """
        return self._request("GET", f"/settings/{service}").json()

    def change_settings(
        self, service: str, values: dict[str, str | None], version: str
    ) -> dict:
        """Changes what one service is configured to do.

        A null value returns that setting to whatever the files say. The
        version is the one the panel was drawn from; a stale one is refused.
        """
        return self._request(
            "PATCH",
            f"/settings/{service}",
            json={"values": values, "version": version},
        ).json()
