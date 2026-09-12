"""Backend calls made by the Documents, Passages, Facts and Topics pages.

One client, because all four ask the same shape of question - a page of rows
about a document - and all four need the document list to choose from.
"""

from __future__ import annotations

from urllib.parse import urlencode

from lib.backend.base import Endpoint


class CatalogApi(Endpoint):
    """What the pages that show the pipeline's output ask the backend."""

    def documents(self, **filters) -> dict:
        """Fetches one page of documents, with the state of each stage."""
        return self._query("/documents", filters)

    def document_names(self) -> list[dict]:
        """Fetches every document by name, for the document picker."""
        return self._get("/documents/names").json()

    def file(self, sha256: str) -> bytes:
        """Fetches one stored document."""
        return self._get(f"/documents/{sha256}/file", timeout=60).content

    def delete(self, sha256: str, derived_only: bool = False) -> dict:
        """Deletes a document, or only what the pipeline built from it."""
        path = f"/documents/{sha256}" + ("/derived" if derived_only else "")
        return self._delete(path).json()

    def stage_status(self, stage: str) -> dict:
        """Reads one stage's queue depth and whether a worker is on it."""
        return self._get(f"/{stage}/status").json()

    def stage_action(self, stage: str, action: str) -> dict:
        """Moves rows on or off one stage's queue.

        None of these runs anything: they move rows between statuses, and the
        stage's worker picks up whatever is claimable on its next poll.
        """
        return self._post(f"/{stage}/{action}", timeout=30).json()

    def topics(self) -> list[dict]:
        """Fetches the fitted topics with how much of the corpus each holds."""
        return self._get("/topics").json()

    def topic_fit(self) -> dict:
        """Fetches the state of the topic model as a whole."""
        return self._get("/topics/fit").json()

    def describe_topic(
        self, topic_id: int, label: str | None, include_in_coverage: bool
    ) -> dict:
        """Names a topic, or takes it out of coverage reporting."""
        return self._patch(
            f"/topics/{topic_id}",
            json={"label": label, "include_in_coverage": include_in_coverage},
        ).json()

    def discover_topics(self) -> dict:
        """Queues a fit over the whole corpus."""
        return self._post("/topics/discover", timeout=30).json()

    def delete_topics(self) -> dict:
        """Deletes every topic and membership."""
        return self._delete("/topics").json()

    def passage_types(self) -> list[str]:
        """Fetches the block types present in the corpus."""
        return self._get("/passages/types").json()

    def passages(self, **filters) -> dict:
        """Fetches one page of passages."""
        return self._query("/passages", filters)

    def passage(self, passage_id: int) -> dict:
        """Fetches one passage in full, its sentences and cell grids included."""
        return self._get(f"/passages/{passage_id}").json()

    def facts(self, **filters) -> dict:
        """Fetches one page of facts."""
        return self._query("/facts", filters)

    def fact_quality(self, **filters) -> dict:
        """Fetches how well extraction is doing, under the same filter."""
        return self._query("/facts/quality", filters)

    def _query(self, path: str, filters: dict) -> dict:
        """Fetches one filtered endpoint, dropping the empty filters."""
        query = urlencode({k: v for k, v in filters.items() if v not in (None, "")})
        return self._get(f"{path}?{query}" if query else path).json()
