"""The driver the topic integration tests act through.

The same idea as a page object: one class holding the SQL and the wiring, so
a test body says what it is about. Everything here runs against the migrated
database the fixtures start.
"""

from __future__ import annotations

from typing import Any

from seed import digest, document, fact, passage
from sqlalchemy import text
from sqlalchemy.orm import Session

from topic_modelling.models import FittedTopic, Fitting, PassageWeight, TopicSpace
from topic_modelling.repository import TopicCatalog, TopicQueue

#: A space the fitter would have produced, small enough to write by hand.
EMPTY_SPACE = TopicSpace(
    topic_term=[], doc_topic=[], doc_lengths=[], vocabulary=[], term_frequency=[]
)


def fitting(
    language: str = "de",
    *,
    topics: list[FittedTopic] | None = None,
    weights: list[PassageWeight] | None = None,
    passages: int = 2,
    without_topics: int = 0,
    vocabulary: int = 5,
) -> Fitting:
    """What one language's fit would have handed the repository."""
    return Fitting(
        language=language,
        topics=topics
        or [
            FittedTopic(0, ["lieferung", "versand"]),
            FittedTopic(1, ["wartung", "reparatur"]),
        ],
        weights=weights or [],
        passages=passages,
        without_topics=without_topics,
        vocabulary=vocabulary,
        space=EMPTY_SPACE,
    )


class TopicStore:
    """The topic tables, as the repositories and a test both see them."""

    def __init__(self, engine) -> None:
        """Binds to the engine the fixtures migrated."""
        self.engine = engine
        self.queue = TopicQueue()
        self.catalog = TopicCatalog()
        self.passages: dict[str, list[int]] = {}

    # ── Setting the corpus up ─────────────────────────────────────────────

    def given_passages(self, **by_language: int) -> TopicStore:
        """Writes one document per language and that many passages in it.

        A passage with no language is asked for as `none`.
        """
        with Session(self.engine) as session:
            for code, count in by_language.items():
                sha = digest(code)
                session.add(document(sha))
                session.flush()
                written = []
                for ordinal in range(1, count + 1):
                    one = passage(
                        sha,
                        ordinal=ordinal,
                        language=None if code == "none" else code,
                        lemmas=["lieferung", "versand"],
                        text=f"{code} passage {ordinal}",
                    )
                    session.add(one)
                    session.flush()
                    written.append(one.id)
                self.passages[code] = written
            session.commit()
        return self

    def given_facts(self, passage_id: int, *, validated: int, rejected: int = 0):
        """Writes facts against one passage."""
        with Session(self.engine) as session:
            for _ in range(validated):
                session.add(fact(passage_id, validated=True))
            for _ in range(rejected):
                session.add(
                    fact(
                        passage_id,
                        validated=False,
                        rejection_code="unsupported_addition",
                    )
                )
            session.commit()
        return self

    def given_table_passage(self, language: str = "de") -> int:
        """Writes one passage that is a table rather than prose."""
        with Session(self.engine) as session:
            sha = digest(f"{language}-table")
            session.add(document(sha))
            session.flush()
            one = passage(
                sha,
                ordinal=1,
                language=language,
                block_type="table",
                lemmas=["besoldungsgruppe", "anzahl"],
            )
            session.add(one)
            session.flush()
            written = one.id
            session.commit()
        self.passages.setdefault(language, []).append(written)
        return written

    # ── Running the queue ────────────────────────────────────────────────

    def request(self) -> int:
        """Asks for a fit."""
        return self.queue.request()

    def claim(self) -> int | None:
        """Takes the next queued fit."""
        return self.queue.claim()

    def store(self, *fittings: Fitting, fit_id: int | None = None) -> int:
        """Replaces every topic with these fittings, claiming first if needed."""
        if fit_id is None:
            fit_id = self.claim() or self.request()
        return self.queue.replace(fit_id, list(fittings))

    def weights_for(self, language: str, *pairs: tuple[int, int, float]):
        """Builds memberships as (passage offset, topic index, weight)."""
        held = self.passages[language]
        return [
            PassageWeight(passage_id=held[offset], topic_index=index, weight=weight)
            for offset, index, weight in pairs
        ]

    # ── Reading it back ──────────────────────────────────────────────────

    def rows(self) -> list[dict[str, Any]]:
        """Every row of `topics`, as a plain mapping."""
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        "SELECT id, language, topic_index, status, label, "
                        "labelled_by, include_in_coverage, error, "
                        "question_status, corpus_passages, corpus_vocabulary, "
                        "passages_without_topics, fitted_at, requested_at "
                        "FROM topics ORDER BY id"
                    )
                ).mappings()
            ]

    def requests(self) -> list[dict[str, Any]]:
        """The rows that are fit requests rather than topics."""
        return [row for row in self.rows() if row["topic_index"] is None]

    def given_passage_vectors(self, **by_passage: int) -> TopicStore:
        """Points each named passage down one axis of the embedding space.

        `given_passage_vectors(**{"1": 0, "2": 0, "3": 7})` puts passages 1
        and 2 on the same axis and 3 on another, so a centroid over the first
        two is that axis and the two centroids are orthogonal.
        """
        with self.engine.connect() as connection:
            for passage_id, axis in by_passage.items():
                vector = [0.0] * 1024
                vector[axis] = 1.0
                connection.execute(
                    text("UPDATE passages SET embedding = :v WHERE id = :id"),
                    {"v": str(vector), "id": int(passage_id)},
                )
            connection.commit()
        return self

    def topic_vectors(self) -> dict[int, list[float] | None]:
        """Each topic's centroid, by topic index."""
        with self.engine.connect() as connection:
            return {
                row.topic_index: (
                    [float(one) for one in str(row.embedding)[1:-1].split(",")]
                    if row.embedding is not None
                    else None
                )
                for row in connection.execute(
                    text(
                        "SELECT topic_index, embedding FROM topics "
                        "WHERE topic_index IS NOT NULL ORDER BY topic_index"
                    )
                )
            }

    def stored_topics(self) -> list[dict[str, Any]]:
        """The rows that are topics."""
        return [row for row in self.rows() if row["topic_index"] is not None]

    def memberships(self) -> list[tuple[int, int, float]]:
        """Every row of `passage_topics`."""
        with self.engine.connect() as connection:
            return [
                (row.passage_id, row.topic_id, row.weight)
                for row in connection.execute(
                    text(
                        "SELECT passage_id, topic_id, weight FROM passage_topics "
                        "ORDER BY passage_id, topic_id"
                    )
                )
            ]

    def topic_id(self, language: str, topic_index: int) -> int:
        """The primary key of one fitted topic."""
        return next(
            row["id"]
            for row in self.stored_topics()
            if row["language"] == language and row["topic_index"] == topic_index
        )

    def delete_passages(self, language: str) -> None:
        """Deletes one language's passages, as a re-chunk would."""
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM passages WHERE language = :code"),
                {"code": language},
            )


class TopicsApi:
    """The /topics routes, as a page would call them."""

    def __init__(self, client) -> None:
        """Binds to the test client."""
        self.client = client

    def status(self) -> dict:
        """Reads the fit queue."""
        return self.client.get("/topics/status").json()

    def topics(self) -> list[dict]:
        """Lists the fitted topics."""
        return self.client.get("/topics").json()

    def fit(self) -> dict:
        """Reads the state of the model as a whole."""
        return self.client.get("/topics/fit").json()

    def discover(self):
        """Queues a fit."""
        return self.client.post("/topics/discover")

    def stop(self):
        """Withdraws a queued fit."""
        return self.client.post("/topics/stop")

    def retry(self):
        """Returns a failed fit to the queue."""
        return self.client.post("/topics/retry")

    def describe(self, topic_id: int, **body):
        """Names a topic, or takes it out of coverage."""
        return self.client.patch(f"/topics/{topic_id}", json=body)

    def visualisation(self, language: str):
        """Fetches one language's figure."""
        return self.client.get(f"/topics/visualisation/{language}")

    def delete(self):
        """Deletes every topic."""
        return self.client.delete("/topics")
