"""Initial schema.

Every table the models declare, as they stood when migrations were adopted.
An existing deployment is brought onto the migration history with
`alembic stamp head` rather than by running this.

Revision: e68b49264a95
Parent:   -
Created:  2026-09-12
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from database.qa_generator.question_facts import (
    DELETE_ORPHAN_FUNCTION,
    DELETE_ORPHAN_TRIGGER,
    DROP_DELETE_ORPHAN,
)

revision: str = "e68b49264a95"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Applies the change."""
    op.create_table(
        "documents",
        sa.Column(
            "sha256",
            sa.CHAR(length=64),
            nullable=False,
            comment="SHA-256 of the raw file bytes. Primary key, and the object key in both buckets: documents/{sha[0:2]}/{sha[2:4]}/{sha} with the extension mime_type implies, and parsed/.../{sha}.json. The 2x2-hex fanout keeps the SeaweedFS filer off a single hot directory. Both keys are derived in code, never stored.",
        ),
        sa.Column(
            "mime_type",
            sa.Text(),
            nullable=False,
            comment="Detected from the leading bytes at upload. The type declared by the upload form is ignored.",
        ),
        sa.Column(
            "page_count",
            sa.Integer(),
            nullable=True,
            comment="Pages, counted by PyMuPDF at upload.",
        ),
        sa.Column(
            "char_count",
            sa.Integer(),
            nullable=True,
            comment="Extractable characters, counted by PyMuPDF at upload. Zero suggests a scanned document.",
        ),
        sa.Column(
            "language",
            sa.CHAR(length=2),
            nullable=True,
            comment="ISO 639-1, NULL until parsing detects it. Constrained to the shape of a code, not to a list of them: which languages the corpus holds is PARSING_LANGUAGES, and the detector can only answer from that set. Note the OCR configuration uses 639-2 codes instead.",
        ),
        sa.Column(
            "title",
            sa.Text(),
            nullable=True,
            comment="Document title, written by parsing.",
        ),
        sa.Column(
            "content_sha256",
            sa.CHAR(length=64),
            nullable=True,
            comment='SHA-256 of the body text - table cell values included - with page_header and page_footer blocks removed, so a re-export of the same page hashes identically despite different bytes while two fee schedules differing only in their numbers do not. Only parsing can produce it, because only Docling classifies blocks by layout. Nothing acts on it at ingest; it answers "do we already have this content?" on request.',
        ),
        sa.Column(
            "parse_confidence",
            sa.Float(),
            nullable=True,
            comment="Mean confidence the parser reported across the document, in [0, 1]. NULL when the parser reported none. The grade Docling derives from this is not stored: its thresholds are Docling's, and a stored grade would disagree with a recomputed one after an upgrade.",
        ),
        sa.Column(
            "parse_confidence_low",
            sa.Float(),
            nullable=True,
            comment="Confidence of the worst page. A document can average well and still have one unreadable page, which is the case worth reviewing, so this is the more useful of the two for finding problems.",
        ),
        sa.Column(
            "parse_status",
            sa.Text(),
            server_default="new",
            nullable=False,
            comment="new | pending | in_progress | parsed | failed. The queue the parsing service selects on; in_progress is how a worker claims a document so a second worker skips it. A worker that dies leaves the row claimed, and the next run fails it with a reason and retry returns it to pending.",
        ),
        sa.Column(
            "parse_error",
            sa.Text(),
            nullable=True,
            comment="Why a failed document failed to parse.",
        ),
        sa.Column(
            "parse_claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When a worker claimed this document for parsing, and NULL whenever it is not claimed. It is what tells a live claim from one a dead worker left behind: without it a second worker starting up cannot sweep abandoned rows without also killing the first worker's work in progress.",
        ),
        sa.Column(
            "chunk_status",
            sa.Text(),
            server_default="new",
            nullable=False,
            comment="new | pending | in_progress | chunked | failed. The queue the chunking service selects on, and the only column it reads to find work: it does not consult parse_status, because a stage knows of no other stage. Starting chunking on a document that was never parsed fails that document, which is the orchestrator's mistake to avoid. Mirrors parse_status in every other respect, including how a dead worker's claim becomes a failure the next run can retry. Extraction has no column here: it queues over passages, because a document is a unit of work that divides very unevenly.",
        ),
        sa.Column(
            "chunk_error",
            sa.Text(),
            nullable=True,
            comment="Why a failed document failed to chunk.",
        ),
        sa.Column(
            "chunk_claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When a worker claimed this document for chunking, and NULL whenever it is not claimed. It is what tells a live claim from one a dead worker left behind: without it a second worker starting up cannot sweep abandoned rows without also killing the first worker's work in progress.",
        ),
        sa.Column(
            "dropped_short",
            sa.Integer(),
            nullable=True,
            comment='Chunks the last run discarded for falling under CHUNKING_MIN_CHARS. Stored rather than logged, because it is content that reached no passage and no fact: "which documents lost content to the size bounds?" is otherwise unanswerable from the database.',
        ),
        sa.Column(
            "dropped_long",
            sa.Integer(),
            nullable=True,
            comment="Chunks the last run discarded for exceeding CHUNKING_MAX_CHARS, typically one large unsplittable table. Counted for the same reason as dropped_short.",
        ),
        sa.CheckConstraint(
            "chunk_status IN ('new', 'pending', 'in_progress', 'failed', 'chunked')",
            name="documents_chunk_status_valid",
        ),
        sa.CheckConstraint(
            "language ~ '^[a-z]{2}$'", name="documents_language_is_iso_639_1"
        ),
        sa.CheckConstraint(
            "parse_status IN ('new', 'pending', 'in_progress', 'failed', 'parsed')",
            name="documents_parse_status_valid",
        ),
        sa.CheckConstraint(
            "parse_confidence BETWEEN 0 AND 1 AND parse_confidence_low BETWEEN 0 AND 1",
            name="documents_parse_confidence_is_a_probability",
        ),
        sa.PrimaryKeyConstraint("sha256"),
        comment="One stored source document. Everything about the act of uploading it - filename, size, timestamp - lives in ingest_events. Ingestion writes sha256, mime_type, page_count and char_count; parsing writes the rest.",
    )
    op.create_index(
        op.f("ix_documents_chunk_status"), "documents", ["chunk_status"], unique=False
    )
    op.create_index(
        op.f("ix_documents_parse_status"), "documents", ["parse_status"], unique=False
    )
    op.create_table(
        "questions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "question_text",
            sa.Text(),
            nullable=False,
            comment="The question as it would be put to the chatbot.",
        ),
        sa.Column(
            "target_answer",
            sa.Text(),
            nullable=True,
            comment="The expected answer. Must be NULL when answerable is false, enforced by a CHECK constraint: an unanswerable question is scored on behaviour, not on content.",
        ),
        sa.Column(
            "answerable",
            sa.Boolean(),
            nullable=False,
            comment="Whether the document supports an answer at all. False for the deliberately unanswerable questions that test whether the chatbot recognises the limits of its knowledge.",
        ),
        sa.Column(
            "difficulty",
            sa.Text(),
            nullable=True,
            comment="Difficulty band, used to stratify the review sample.",
        ),
        sa.Column(
            "language",
            sa.CHAR(length=2),
            nullable=False,
            comment="ISO 639-1 language the question is written in, constrained to the shape of a code rather than to a list of them. Deliberately independent of the source document's language: a German document can carry English questions.",
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=768),
            nullable=True,
            comment="Question embedding, used by the semantic dedup and unanswerability gates. Those do cosine search over every accepted question and will need an HNSW index before they are usable at scale. 768 dimensions because that is what most open embedding models emit - nomic-embed-text, bge-base, e5-base, all-mpnet-base-v2, gte-base. A model with a wider native output, such as Qwen3-Embedding at 4096, must be truncated to fit, which its Matryoshka training supports.",
        ),
        sa.Column(
            "status",
            sa.Text(),
            server_default="draft",
            nullable=False,
            comment="draft | accepted | rejected, enforced by a CHECK constraint. Rejected questions are never deleted: they are the drop-rate evidence in the coverage report.",
        ),
        sa.Column(
            "rejected_reason",
            sa.Text(),
            nullable=True,
            comment="Which gate rejected the question.",
        ),
        sa.Column(
            "holdout_set_id",
            sa.Uuid(),
            nullable=True,
            comment="The hold-out draw this question belongs to; one UUID per release. Non-NULL means the question is withheld from the open export and kept encrypted, so there is a test the development team has not tuned against. NULL means it is in the open set.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the question was generated.",
        ),
        sa.Column(
            "status_changed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When status last changed.",
        ),
        sa.CheckConstraint(
            "language ~ '^[a-z]{2}$'", name="questions_language_is_iso_639_1"
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'accepted', 'rejected')", name="questions_status_valid"
        ),
        sa.CheckConstraint(
            "answerable OR target_answer IS NULL",
            name="questions_unanswerable_has_no_target",
        ),
        sa.PrimaryKeyConstraint("id"),
        comment="The test questions themselves. Append-only and never hard-deleted: rejected questions are the drop-rate evidence in the coverage report. Deleting every fact a question came from also deletes the question: a foreign key cannot cascade that direction, so an AFTER DELETE trigger on question_facts does it.",
    )
    op.create_index(
        "ix_questions_embedding_hnsw",
        "questions",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_table(
        "topics",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "topic_index",
            sa.Integer(),
            nullable=True,
            comment="Position of this topic in the fitted model, and the mapping key back to it: scoring a passage returns an index, and this turns that index into a row. Unique, because two rows sharing an index would make that mapping ambiguous. NULL means this row is an outstanding fit request rather than a topic; NULLs do not collide under a unique constraint.",
        ),
        sa.Column(
            "top_terms",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            comment="Highest-weighted terms, in descending order. The human-readable identity of the topic, and the only stored signature available for matching a topic across refits. Empty only on a row that is a fit request rather than a topic.",
        ),
        sa.Column(
            "label",
            sa.Text(),
            nullable=True,
            comment='Name assigned by a person, if any, so reports read "Account fees" rather than a list of terms. Carried across a refit by matching top terms, which is the only signature there is; a topic whose terms have moved too far loses it.',
        ),
        sa.Column(
            "include_in_coverage",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
            comment="Whether this topic counts toward coverage reporting. Set false for topics judged too unstable or too diffuse to be a meaningful coverage partition. Carried across a refit with the label.",
        ),
        sa.Column(
            "status",
            sa.Text(),
            server_default="pending",
            nullable=False,
            comment="pending | in_progress | modelled | failed. A finished topic is modelled; the other three describe a fit rather than a topic. in_progress is how a worker claims a request so a second worker skips it, and a worker that dies leaves the row claimed until the lease expires and the next run fails it with a reason.",
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
            comment="Why a fit failed. On a failed request row, which is kept beside the topics the previous fit produced - a refit that fails does not take the working model with it.",
        ),
        sa.Column(
            "claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When a worker claimed this fit, and NULL whenever it is not claimed. It is what tells a live claim from one a dead worker left behind.",
        ),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the fit that produced this topic was asked for.",
        ),
        sa.Column(
            "fitted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the fit completed, and NULL until it does. The answer to whether the stored topics still describe the corpus as it is now: compare it against the newest document.",
        ),
        sa.Column(
            "corpus_passages",
            sa.Integer(),
            nullable=True,
            comment="How many passages the model was fitted over. Repeated on every topic of one fit, because it qualifies each of them: twelve topics over thirty passages are noise, and the number is what says so.",
        ),
        sa.Column(
            "corpus_vocabulary",
            sa.Integer(),
            nullable=True,
            comment="How many terms survived the frequency filter and so define the topics. A collapse here is the first sign that TOPIC_NO_BELOW or TOPIC_NO_ABOVE is wrong for the corpus.",
        ),
        sa.Column(
            "passages_without_topics",
            sa.Integer(),
            nullable=True,
            comment="How many passages the fit could place in no topic at all, because every one of their terms was filtered out of the vocabulary. A coverage gap: those passages, and the facts and questions drawn from them, are absent from any topic-weighted report. Counted rather than logged, for the same reason a rejected fact is stored rather than dropped.",
        ),
        sa.CheckConstraint(
            "status <> 'modelled' OR (topic_index IS NOT NULL AND cardinality(top_terms) > 0)",
            name="topics_modelled_is_a_topic",
        ),
        sa.CheckConstraint(
            "status IN ('new', 'pending', 'in_progress', 'failed', 'modelled')",
            name="topics_status_valid",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("topic_index", name="topics_topic_index_unique"),
        comment="Clusters over the corpus vocabulary, fitted by LDA. Not an LLM and not pretrained: topics are defined over this corpus's own vocabulary, which is what keeps the system industry-agnostic. Standalone by design - it references nothing. passage_topics carries the weighted membership, and facts and questions reach their topics by joining through their passage, so no two rows can disagree about it. Deleting a topic removes its membership rows only. This is also the queue: a row with a NULL topic_index is an outstanding request to refit, not a topic, and a completed fit replaces every row - LDA cannot refit one topic in isolation, so there is no per-topic rediscovery.",
    )
    op.create_index(op.f("ix_topics_status"), "topics", ["status"], unique=False)
    op.create_table(
        "ingest_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "submitted_filename",
            sa.Text(),
            nullable=False,
            comment="Filename as submitted. The earliest stored event for a hash is that document's original filename.",
        ),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the upload was received.",
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
            comment="Size of the submitted file.",
        ),
        sa.Column(
            "sha256",
            sa.CHAR(length=64),
            nullable=True,
            comment="SHA-256 of the submitted bytes, and the link to the document they became. NULL for uploads refused before hashing, which is why the foreign key tolerates NULL.",
        ),
        sa.Column(
            "outcome",
            sa.Text(),
            nullable=False,
            comment="stored | duplicate_bytes | too_large | unsupported_type. Enforced by a CHECK constraint.",
        ),
        sa.Column(
            "detail",
            sa.Text(),
            nullable=True,
            comment="Human-readable reason, shown to the uploader when a file is refused.",
        ),
        sa.CheckConstraint(
            "outcome IN ('stored', 'duplicate_bytes', 'too_large', 'unsupported_type')",
            name="ingest_events_outcome_valid",
        ),
        sa.ForeignKeyConstraint(["sha256"], ["documents.sha256"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        comment="Every upload attempt, including refused ones. A refused upload writes nothing to documents, so this is the only record that it happened. Also holds ingest provenance: the same bytes arriving three times under three names is one documents row and three rows here.",
    )
    op.create_table(
        "passages",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "doc_sha256",
            sa.CHAR(length=64),
            nullable=False,
            comment="The document this passage was cut from.",
        ),
        sa.Column(
            "ordinal",
            sa.Integer(),
            nullable=False,
            comment="Position in reading order within the document, starting at 1 and contiguous. Assigned by the chunker following the order of Docling's structured model, not raw PDF byte order. Contiguity is what makes ordinal +/- 1 the neighbouring passages, which the edge-case generator needs. Reassigned on every re-chunk, so it is not stable across chunking configuration changes.",
        ),
        sa.Column(
            "extract_status",
            sa.Text(),
            server_default="new",
            nullable=False,
            comment="new | pending | in_progress | extracted | failed. The extraction queue runs over passages rather than documents: one document holds four hundred passages and another thirty, so claiming a document hands one worker hours of work and its neighbour minutes. A passage is one model call, which makes it the unit that divides evenly.",
        ),
        sa.Column(
            "extract_error",
            sa.Text(),
            nullable=True,
            comment="Why this passage could not be read.",
        ),
        sa.Column(
            "extract_claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When a worker claimed this passage, NULL when none holds it.",
        ),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
            comment="The passage content. Evidence offsets on facts are relative to this string.",
        ),
        sa.Column(
            "page_from",
            sa.Integer(),
            nullable=True,
            comment="Page the passage starts on.",
        ),
        sa.Column(
            "page_to",
            sa.Integer(),
            nullable=True,
            comment="Page the passage ends on. Differs from page_from when the passage crosses a page break.",
        ),
        sa.Column(
            "section_path",
            sa.Text(),
            nullable=True,
            comment='Heading trail, for example "3 > 3.2 > Fees". What a reviewer reads to orient themselves.',
        ),
        sa.Column(
            "block_type",
            sa.Text(),
            nullable=True,
            comment="Docling's DocItemLabel for the source item: text, section_header, list_item, table, caption, formula and others. Routes extraction: a table passage goes to the deterministic cell reader, everything else to the LLM. A passage holding items of several kinds is labelled table if any of them is one, because reading a table with the model is the more expensive mistake.",
        ),
        sa.Column(
            "doc_item_refs",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            comment="The converter's own identifiers for the items this passage was cut from, for example {#/texts/28,#/texts/31}. The only way back from a stored passage to the converted document in the parsed bucket: without them the mapping is fuzzy text matching, which fails on exactly the repeated boilerplate where it matters. Also the record of what a merged passage is made of, which block_type alone cannot express.",
        ),
        sa.Column(
            "bbox",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
            comment="Where the passage sits on the page, one box per page it spans: [{page, l, t, r, b}]. Normalised to a top-left origin, because the converter reports text from the bottom left and table cells from the top left, and a consumer cannot be expected to know which. Stored so that drawing a highlight over the source does not mean fetching and parsing a multi-megabyte document from the object store. NULL when the converter recorded no position.",
        ),
        sa.Column(
            "table_cells",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
            comment="The cell grid behind a table passage. A list of grids, since one passage can hold more than one table, each {caption, num_rows, num_cols, cells:[{row, col, row_span, col_span, column_header, row_header, text, line}]}. NULL unless block_type is table. `line` is the rendered Markdown row of passages.text the cell sits in, and the evidence any fact drawn from that cell quotes; it is NULL on a header cell, which yields no fact. Only the rows this passage renders are stored: a split table points every piece at the whole item, and a row this piece does not show would describe a value with a row the reader cannot see.",
        ),
        sa.CheckConstraint(
            "extract_status IN ('new', 'pending', 'in_progress', 'failed', 'extracted')",
            name="passages_extract_status_valid",
        ),
        sa.CheckConstraint("ordinal > 0", name="passages_ordinal_positive"),
        sa.ForeignKeyConstraint(
            ["doc_sha256"], ["documents.sha256"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_sha256", "ordinal"),
        comment="A document split into retrievable chunks. Immutable: re-chunking deletes and re-inserts every passage of a document, which cascades to its facts and questions.",
    )
    op.create_index(
        op.f("ix_passages_extract_status"), "passages", ["extract_status"], unique=False
    )
    op.create_index(
        "ix_passages_text_trgm",
        "passages",
        ["text"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"text": "gin_trgm_ops"},
    )
    op.create_table(
        "facts",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "passage_id",
            sa.BigInteger(),
            nullable=False,
            comment="The passage this fact was extracted from, and the route to both its document and its topic. A fact has no topic column of its own: one value in one place cannot disagree with itself.",
        ),
        sa.Column(
            "statement",
            sa.Text(),
            nullable=False,
            comment="The fact as a single self-contained sentence.",
        ),
        sa.Column(
            "evidence_text",
            sa.Text(),
            nullable=False,
            comment="The exact source wording supporting the statement. The evidence_exact gate matches this against the passage text character for character.",
        ),
        sa.Column(
            "evidence_start",
            sa.Integer(),
            nullable=False,
            comment="Offset of the evidence's first character, RELATIVE TO passages.text - not to the page and not to the document.",
        ),
        sa.Column(
            "evidence_end",
            sa.Integer(),
            nullable=False,
            comment="Offset one past the evidence's last character, relative to passages.text, so the span is text[evidence_start:evidence_end] and its length is the difference. Present so that evidence appearing twice in one passage is still unambiguous.",
        ),
        sa.Column(
            "evidence_occurrences",
            sa.Integer(),
            server_default="1",
            nullable=False,
            comment="How many times the evidence appears in the passage. One means the stored span is the only candidate and the citation is exact. More means the span is the FIRST of several identical ones and may not be the copy the fact was drawn from - nothing in a model's answer says which, so the ambiguity is recorded rather than guessed at. Filter on this when a citation has to be unambiguous.",
        ),
        sa.Column(
            "extraction_method",
            sa.Text(),
            nullable=False,
            comment="llm | deterministic. Deterministic facts are read straight from table cells and carry no model-error risk; llm facts do. Enforced by a CHECK constraint.",
        ),
        sa.Column(
            "extraction_model",
            sa.Text(),
            nullable=True,
            comment="The model that produced this fact, as the LiteLLM identifier it was called by, and NULL for a deterministic one. Without it the table silently mixes generations: change EXTRACTION_MODEL and there is no way to tell which facts came from which model, which for a deliverable that is itself a reference dataset makes the whole table unciteable.",
        ),
        sa.Column(
            "prompt_version",
            sa.Text(),
            nullable=True,
            comment="Version of the extraction prompt, NULL for a deterministic fact. The prompt decides what counts as a fact, so two prompts are two datasets; this is what lets them be told apart.",
        ),
        sa.Column(
            "extraction_temperature",
            sa.Float(),
            nullable=True,
            comment="Sampling temperature used, NULL for a deterministic fact. Recorded because a fact drawn at temperature 0 is reproducible and one drawn above it is not.",
        ),
        sa.Column(
            "validated",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
            comment="Whether the evidence was found in the passage.",
        ),
        sa.Column(
            "validation_error",
            sa.Text(),
            nullable=True,
            comment="Why validation failed.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the fact was extracted.",
        ),
        sa.CheckConstraint(
            "extraction_method IN ('llm', 'deterministic')",
            name="facts_extraction_method_valid",
        ),
        sa.CheckConstraint(
            "evidence_end >= evidence_start", name="facts_evidence_span_ordered"
        ),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        comment="One atomic statement isolated from one passage, with the exact source span supporting it. The unit a question is generated from and scored against. Deliberately industry-agnostic: there are no typed domain entities here, so nothing in this table assumes banking.",
    )
    op.create_index(
        "ix_facts_evidence_trgm",
        "facts",
        ["evidence_text"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"evidence_text": "gin_trgm_ops"},
    )
    op.create_index(op.f("ix_facts_passage_id"), "facts", ["passage_id"], unique=False)
    op.create_index(
        "ix_facts_statement_trgm",
        "facts",
        ["statement"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"statement": "gin_trgm_ops"},
    )
    op.create_table(
        "passage_topics",
        sa.Column(
            "passage_id", sa.BigInteger(), nullable=False, comment="The passage."
        ),
        sa.Column(
            "topic_id",
            sa.BigInteger(),
            nullable=False,
            comment="The topic. Deleting a topic removes these membership rows and leaves every passage in place, holding one fewer topic.",
        ),
        sa.Column(
            "weight",
            sa.Float(),
            nullable=False,
            comment="Share of this passage attributed to this topic by the model, in (0, 1], enforced by a CHECK constraint. The weights for one passage SHOULD sum to at most 1, but that is NOT enforced: it spans rows, so no row-level constraint can see it and only a trigger could. The writer is trusted, because the only writer is the topic model, which produces a normalised distribution by construction.",
        ),
        sa.CheckConstraint(
            "weight > 0 AND weight <= 1", name="passage_topics_weight_is_a_probability"
        ),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("passage_id", "topic_id"),
        comment="How strongly one passage belongs to one topic - the LDA distribution, one row per pair. The only place the passage-to-topic relationship is stored; facts and questions reach their topics by joining through their passage. The dominant topic is the highest weight for a passage, derived rather than stored.",
    )
    op.create_table(
        "question_facts",
        sa.Column(
            "question_id", sa.BigInteger(), nullable=False, comment="The question."
        ),
        sa.Column(
            "fact_id",
            sa.BigInteger(),
            nullable=False,
            comment="A fact the question tests. Deleting the fact removes this row; see the note on the questions table about questions left with no facts.",
        ),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("question_id", "fact_id"),
        comment="Which facts a question was generated from. One row for an ordinary question, two or more for a cross-document question. The passages a question needs retrieved are derived from here through facts.passage_id rather than stored separately.",
    )
    op.create_index(
        op.f("ix_question_facts_fact_id"), "question_facts", ["fact_id"], unique=False
    )

    # Not autogenerated: Alembic compares tables, columns and indexes, and
    # sees no trigger. See the note in database/qa_generator/question_facts.
    op.execute(DELETE_ORPHAN_FUNCTION)
    op.execute(DELETE_ORPHAN_TRIGGER)


def downgrade() -> None:
    """Takes it back out."""
    op.execute(DROP_DELETE_ORPHAN)
    op.drop_index(op.f("ix_question_facts_fact_id"), table_name="question_facts")
    op.drop_table("question_facts")
    op.drop_table("passage_topics")
    op.drop_index(
        "ix_facts_statement_trgm",
        table_name="facts",
        postgresql_using="gin",
        postgresql_ops={"statement": "gin_trgm_ops"},
    )
    op.drop_index(op.f("ix_facts_passage_id"), table_name="facts")
    op.drop_index(
        "ix_facts_evidence_trgm",
        table_name="facts",
        postgresql_using="gin",
        postgresql_ops={"evidence_text": "gin_trgm_ops"},
    )
    op.drop_table("facts")
    op.drop_index(
        "ix_passages_text_trgm",
        table_name="passages",
        postgresql_using="gin",
        postgresql_ops={"text": "gin_trgm_ops"},
    )
    op.drop_index(op.f("ix_passages_extract_status"), table_name="passages")
    op.drop_table("passages")
    op.drop_table("ingest_events")
    op.drop_index(op.f("ix_topics_status"), table_name="topics")
    op.drop_table("topics")
    op.drop_index(
        "ix_questions_embedding_hnsw",
        table_name="questions",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_table("questions")
    op.drop_index(op.f("ix_documents_parse_status"), table_name="documents")
    op.drop_index(op.f("ix_documents_chunk_status"), table_name="documents")
    op.drop_table("documents")
