"""What each service is configured to do, read and changed.

Its own client rather than calls on a page's: six pages draw a
configuration panel and two of them - Upload and System health - hold no
catalogue client at all, so these would otherwise be reached through a
client named for somebody else's page.
"""

from __future__ import annotations

from lib.backend.base import Endpoint


class SettingsApi(Endpoint):
    """The settings routes, one service at a time."""

    def settings(self, service: str) -> dict:
        """Fetches what one service is configured to do.

        Everything a control needs comes back with it - the type, the bounds,
        the closed set of values where there is one - so a page draws its
        panel from this and holds no list of settings of its own.
        """
        return self._request("GET", f"/settings/{service}").json()

    def change_settings(
        self, service: str, values: dict[str, str | None], version: str
    ) -> dict:
        """Changes what one service is configured to do.

        A null value returns that setting to whatever the files say. The
        version is the one the panel was drawn from, so a save that would
        land on somebody else's change is refused rather than overwriting it.
        """
        return self._request(
            "PATCH",
            f"/settings/{service}",
            json={"values": values, "version": version},
        ).json()
