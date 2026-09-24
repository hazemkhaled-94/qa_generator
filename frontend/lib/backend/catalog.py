"""Backend calls made by every page that shows what the pipeline produced.

One client, because they all ask the same shape of question - a page of rows
about a document - and they all need the document list to choose from.
"""

from __future__ import annotations

from urllib.parse import quote, urlencode

from lib.backend.base import Endpoint


def _within(scope: tuple[str, str] | None) -> str:
    """Renders the path segment that narrows a stage route to one item."""
    if scope is None:
        return ""
    kind, value = scope
    return f"/{quote(str(kind))}/{quote(str(value))}"


class CatalogApi(Endpoint):
    """What the pages that show the pipeline's output ask the backend."""

    def documents(self, **filters) -> dict:
        """Fetches one page of documents, with the state of each stage."""
        return self._query("/documents", filters)

    def document_names(self) -> list[dict]:
        """Fetches every document by name, for the document picker."""
        return self._request("GET", "/documents/names").json()

    def file(self, sha256: str) -> bytes:
        """Fetches one stored document."""
        return self._request("GET", f"/documents/{sha256}/file", timeout=60).content

    def delete(self, sha256: str, derived_only: bool = False) -> dict:
        """Deletes a document, or only what the pipeline built from it."""
        path = f"/documents/{sha256}" + ("/derived" if derived_only else "")
        return self._request("DELETE", path).json()

    def stage_status(self, stage: str, scope: tuple[str, str] | None = None) -> dict:
        """Reads one stage's queue depth and whether a worker is on it.

        `scope` narrows it to one item, as the pair the route takes:
        ("document", sha256) or ("passage", id). Without one, the whole
        queue.
        """
        return self._request("GET", f"/{stage}{_within(scope)}/status").json()

    def stage_action(
        self, stage: str, action: str, scope: tuple[str, str] | None = None
    ) -> dict:
        """Moves rows on or off one stage's queue, for one item or for all.

        None of these runs anything: they move rows between statuses, and the
        stage's worker picks up whatever is claimable on its next poll.
        """
        return self._request(
            "POST", f"/{stage}{_within(scope)}/{action}", timeout=30
        ).json()

    def topics(self) -> list[dict]:
        """Fetches the fitted topics with how much of the corpus each holds."""
        return self._request("GET", "/topics").json()

    def topic_fit(self) -> dict:
        """Fetches the state of the topic model as a whole."""
        return self._request("GET", "/topics/fit").json()

    def topic_visualisation(self, language: str) -> str | None:
        """Fetches one language's pyLDAvis page, or None if no fit drew it."""
        response = self._session.get(
            f"{self._base_url}/topics/visualisation/{quote(language)}", timeout=60
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text

    def describe_topic(
        self, topic_id: int, label: str | None, include_in_coverage: bool
    ) -> dict:
        """Names a topic, or takes it out of coverage reporting."""
        return self._request(
            "PATCH",
            f"/topics/{topic_id}",
            json={"label": label, "include_in_coverage": include_in_coverage},
        ).json()

    def discover_topics(self) -> dict:
        """Queues a fit over the whole corpus."""
        return self._request("POST", "/topics/discover", timeout=30).json()

    def delete_topics(self) -> dict:
        """Deletes every topic and membership."""
        return self._request("DELETE", "/topics").json()

    def passage_types(self) -> list[str]:
        """Fetches the block types present in the corpus."""
        return self._request("GET", "/passages/types").json()

    def passages(self, **filters) -> dict:
        """Fetches one page of passages."""
        return self._query("/passages", filters)

    def passage(self, passage_id: int) -> dict:
        """Fetches one passage in full, its sentences and cell grids included."""
        return self._request("GET", f"/passages/{passage_id}").json()

    def facts(self, **filters) -> dict:
        """Fetches one page of facts."""
        return self._query("/facts", filters)

    def fact_quality(self, **filters) -> dict:
        """Fetches how well extraction is doing, under the same filter."""
        return self._query("/facts/quality", filters)

    def questions(self, **filters) -> dict:
        """Fetches one page of questions."""
        return self._query("/questions", filters)

    def question(self, question_id: int) -> dict:
        """Fetches one question with the facts it was written from."""
        return self._request("GET", f"/questions/{question_id}").json()

    def prompts(self, **filters) -> list[dict]:
        """Fetches the recorded prompts, narrowed by service, version or name.

        A list and not a page: there are a few dozen of them and they do
        not grow with the corpus.
        """
        return self._request("GET", "/prompts", params=filters).json()

    def question_quality(self, **filters) -> dict:
        """Fetches how well generation is doing, under the same filter."""
        return self._query("/questions/quality", filters)

    def question_plan(self) -> dict:
        """Fetches what generation is configured to write."""
        return self._request("GET", "/questions/plan").json()

    def question_workbook(self, **filters) -> bytes:
        """Fetches the questions this filter selects as an .xlsx workbook.

        Bytes, not JSON: the API builds the workbook, because the rows it
        holds are the rows a page never asked for - a download is the whole
        filter and a page is fifty of it.
        """
        query = urlencode({k: v for k, v in filters.items() if v not in (None, "")})
        return self._request(
            "GET", f"/questions/export?{query}" if query else "/questions/export"
        ).content

    def decide_question(self, question_id: int, status: str) -> dict:
        """Accepts or rejects one question."""
        return self._request(
            "PATCH", f"/questions/{question_id}", json={"status": status}
        ).json()

    def _query(self, path: str, filters: dict) -> dict:
        """Fetches one filtered endpoint, dropping the empty filters."""
        query = urlencode({k: v for k, v in filters.items() if v not in (None, "")})
        return self._request("GET", f"{path}?{query}" if query else path).json()
