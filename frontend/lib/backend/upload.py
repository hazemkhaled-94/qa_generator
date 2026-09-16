"""Backend calls made by the Upload page."""

from __future__ import annotations

from lib.backend.base import Endpoint


class UploadApi(Endpoint):
    """What the Upload page asks the backend to do."""

    def counts(self) -> dict[str, int]:
        """Reads what ingestion holds, for the page's figures."""
        held = self._request("GET", "/status").json().get("ingestion") or {}
        return held.get("metrics") or {}

    def add_document(self, filename: str, data: bytes) -> dict:
        """Sends one file to the backend.

        Raises:
            requests.exceptions.RequestException: On a network failure, or a
                413 or 415 when the file is refused, with the reason in the
                response body.
        """
        return self._request(
            "POST",
            "/documents",
            files={"file": (filename, data, "application/octet-stream")},
        ).json()
