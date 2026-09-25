"""The one call the lineage panel makes."""

from __future__ import annotations

from urllib.parse import quote

from lib.backend.base import Endpoint


class LineageApi(Endpoint):
    """How one artefact was produced."""

    def lineage(self, kind: str, artifact_id: str) -> dict:
        """Fetches one artefact's chain, in pipeline order."""
        return self._request(
            "GET", f"/lineage/{quote(kind)}/{quote(str(artifact_id))}"
        ).json()
