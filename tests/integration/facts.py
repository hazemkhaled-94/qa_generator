"""The driver the facts integration tests act through.

The same idea as a page object: one class holding the SQL and the wiring, so
a test body says what it is about. Everything here runs against the migrated
database the fixtures start.
"""

from __future__ import annotations

from typing import Any

from seed import digest, document, passage
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.qa_generator import FactKind, PassageTopic, Status, Topic
from extraction.models import CheckedFact, Citation, PassageToExtract
from extraction.repository import FactCatalog, PassageQueue


def checked(
    passage_id: int,
    statement: str = "The device weighs 4 kg.",
    *,
    kind: str = FactKind.ATOMIC,
    validated: bool = True,
    rejection_code: str | None = None,
    passage_ids: list[int] | None = None,
    citations: list[Citation] | None = None,
    **columns: Any,
) -> CheckedFact:
    """One fact as the checker would hand it to the repository.

    Args:
        passage_id: The passage it is anchored to.
        statement: What it says.
        kind: Which reading it is.
        validated: Whether it passed.
        rejection_code: Why it did not, when it did not.
        passage_ids: The passages a bridge rests on, each cited at its
            sentence 0 over an empty span. For a test that reads the links
            rather than the spans.
        citations: The citations themselves, for a test that reads a span.
        **columns: Anything else `CheckedFact` takes.

    Returns:
        The fact.
    """
    return CheckedFact(
        passage_id=passage_id,
        statement=statement,
        evidence_text=columns.pop("evidence_text", statement),
        evidence_sentence_ids=columns.pop("evidence_sentence_ids", [0]),
        evidence_start=columns.pop("evidence_start", 0),
        evidence_end=columns.pop("evidence_end", len(statement)),
        extraction_method=columns.pop("extraction_method", "llm"),
        validated=validated,
        rejection_code=rejection_code,
        validation_error=columns.pop("validation_error", None),
        kind=kind,
        citations=citations
        if citations is not None
        else [
            Citation(passage_id=one, sentence_ids=[0], start=0, end=0)
            for one in (passage_ids or [])
        ],
        **columns,
    )


class FactStore:
    """Writes the rows a facts test needs and reads back what it produced."""

    def __init__(self, engine, language: str = "en") -> None:
        """Initialises the driver against the migrated database.

        Args:
            engine: The engine the fixtures built.
            language: The language every seeded document is in.
        """
        self.engine = engine
        self.language = language
        self.queue = PassageQueue()
        self.catalog = FactCatalog()

    def corpus(self, *documents: tuple[str, list[str]]) -> dict[str, list[int]]:
        """Writes documents and their passages.

        Args:
            *documents: Each document's seed and the text of its passages, in
                reading order.

        Returns:
            The passage ids written, by document seed.
        """
        written: dict[str, list[int]] = {}
        with Session(self.engine) as session:
            for seed, texts in documents:
                sha = digest(seed)
                session.add(document(sha, language=self.language, title=seed))
                session.flush()
                written[seed] = []
                for ordinal, body in enumerate(texts, start=1):
                    row = passage(
                        sha,
                        ordinal,
                        text=body,
                        language=self.language,
                        sentences=[
                            {"i": 0, "start": 0, "end": len(body), "predicates": 1}
                        ],
                    )
                    session.add(row)
                    session.flush()
                    written[seed].append(row.id)
            session.commit()
        return written

    def queued(self, *passage_ids: int) -> None:
        """Marks passages claimable, which is what `start` does."""
        self._execute(
            "UPDATE passages SET extract_status = :status WHERE id = ANY(:ids)",
            status=Status.PENDING,
            ids=list(passage_ids),
        )

    def topic(self, name: str, *passage_ids: int, weight: float = 0.9) -> int:
        """Puts passages in one topic, which is what groups them.

        Args:
            name: What to label it.
            *passage_ids: The passages whose strongest topic this is.
            weight: How strongly each belongs to it.

        Returns:
            The topic's id.
        """
        with Session(self.engine) as session:
            row = Topic(
                language=self.language,
                topic_index=session.query(Topic).count(),
                status="modelled",
                label=name,
                top_terms=["support", "response"],
            )
            session.add(row)
            session.flush()
            for passage_id in passage_ids:
                session.add(
                    PassageTopic(passage_id=passage_id, topic_id=row.id, weight=weight)
                )
            session.commit()
            return row.id

    def store(self, passage_id: int, *facts: CheckedFact) -> int:
        """Writes one passage's facts, as the per-passage service does."""
        return self.queue.store(passage_id, list(facts))

    def bridges(self, *facts: CheckedFact) -> int:
        """Writes bridge facts, as the bridge pass does."""
        return self.catalog.add_bridges(list(facts))

    def rows(self, *columns: str, where: str = "TRUE", **values: Any) -> list[tuple]:
        """Reads columns out of the facts table."""
        return self._execute(
            f"SELECT {', '.join(columns)} FROM facts WHERE {where} ORDER BY id",
            **values,
        )

    def links(self) -> list[tuple]:
        """Every fact_passages row, in the order a bridge was shown them."""
        return self._execute(
            "SELECT fact_id, passage_id, position FROM fact_passages "
            "ORDER BY fact_id, position"
        )

    def citations(self) -> list[tuple]:
        """Where in each passage every bridge rests, in the same order."""
        return self._execute(
            "SELECT passage_id, sentence_ids, evidence_start, evidence_end "
            "FROM fact_passages ORDER BY fact_id, position"
        )

    def cited_text(self) -> list[str]:
        """The text each citation resolves to, read out of its own passage."""
        return [
            row[0]
            for row in self._execute(
                "SELECT substring(p.text FROM fp.evidence_start + 1 FOR "
                "fp.evidence_end - fp.evidence_start) FROM fact_passages fp "
                "JOIN passages p ON p.id = fp.passage_id "
                "ORDER BY fp.fact_id, fp.position"
            )
        ]

    def count(self, table: str = "facts") -> int:
        """How many rows a table holds."""
        return self._execute(f"SELECT count(*) FROM {table}")[0][0]

    def statuses(self) -> list[str]:
        """Every passage's extract status, in id order."""
        return [
            row[0]
            for row in self._execute("SELECT extract_status FROM passages ORDER BY id")
        ]

    def drop_passage(self, passage_id: int) -> None:
        """Deletes a passage, as re-chunking does."""
        self._execute("DELETE FROM passages WHERE id = :id", id=passage_id)

    def group_of(self, topic_id: int) -> list[PassageToExtract]:
        """The passages one topic holds, as the bridge pass reads them."""
        return next(
            (
                passages
                for found, passages in self.catalog.by_topic()
                if found == topic_id
            ),
            [],
        )

    def _execute(self, sql: str, **values: Any) -> list[tuple]:
        """Runs one statement and returns whatever it selected."""
        with self.engine.begin() as connection:
            result = connection.execute(text(sql), values)
            return list(result.all()) if result.returns_rows else []
