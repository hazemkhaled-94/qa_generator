"""The topics table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.status import Status, check, queued

if TYPE_CHECKING:
    from database.qa_generator.passage_topics import PassageTopic


class Topic(Base):
    """One cluster over one language's vocabulary.

    Also the queue: a row whose topic_index is NULL is an outstanding request
    to fit rather than a topic. A fit is always a whole language, because the
    factorisation estimates every topic together over one vocabulary, so a
    completed run replaces every row here.
    """

    # No foreign key anywhere: a topic spans passages in many documents.
    __tablename__ = "topics"
    __table_args__ = (
        # Per language: each language is fitted on its own vocabulary, so an
        # index identifies a topic only together with the language it is in.
        UniqueConstraint("language", "topic_index", name="topics_topic_index_unique"),
        CheckConstraint(check("status", Status.MODELLED), name="topics_status_valid"),
        CheckConstraint(
            "status <> 'modelled' OR (topic_index IS NOT NULL AND "
            "language IS NOT NULL AND cardinality(top_terms) > 0)",
            name="topics_modelled_is_a_topic",
        ),
        # Partial, as on documents and passages: the queue is a shrinking
        # fraction of the table.
        Index(
            "ix_topics_question_queue",
            "question_status",
            postgresql_where=text(queued("question_status")),
        ),
        CheckConstraint(
            check("question_status", Status.GENERATED),
            name="topics_question_status_valid",
        ),
        {
            "comment": "Clusters over one language's own vocabulary, factorised out "
            "of a tf-idf weighted term matrix. "
            "passage_topics carries the membership; facts and questions reach "
            "their topics by joining through their passage. Also two queues: a row "
            "with a NULL topic_index is a request to refit, and a row that is a "
            "topic is a unit of question generation."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    language: Mapped[str | None] = mapped_column(
        CHAR(2),
        comment="Which language's model this topic belongs to. Each language is "
        "fitted separately over its own vocabulary, because one model across "
        "both spends topics on telling the languages apart rather than on "
        "telling subjects apart. NULL on a fit request.",
    )
    topic_index: Mapped[int | None] = mapped_column(
        Integer,
        comment="Position of this topic in its language's fitted model, and the "
        "key scoring a passage returns. Unique per language, not overall. NULL "
        "means this row is a fit request rather than a topic.",
    )
    top_terms: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        comment="Highest-weighted terms, descending. The topic's readable identity "
        "and the only signature a label is matched on across a refit.",
    )
    label: Mapped[str | None] = mapped_column(
        Text, comment="The subject this topic is, in words."
    )
    labelled_by: Mapped[str | None] = mapped_column(
        Text,
        comment="What named it: 'person', or the model identifier. A name a "
        "person typed is never overwritten by a model's, and a report that "
        "reads by name should say which it is reading.",
    )
    include_in_coverage: Mapped[bool] = mapped_column(
        Boolean,
        server_default=true(),
        comment="Whether this topic counts toward coverage reporting. Carried "
        "across a refit with the label.",
    )
    status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.PENDING,
        index=True,
        comment="pending | in_progress | modelled | failed. A finished topic is "
        "modelled; the other three describe a fit.",
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        comment="Why a fit failed, on the request row, which is kept beside the "
        "topics the previous fit produced.",
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When a worker claimed this fit, NULL when none holds it.",
    )
    question_status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.NEW,
        comment="new | pending | in_progress | failed | generated, enforced by a "
        "CHECK constraint. Question generation's own queue, over topics rather "
        "than over fits: the unit of work is one topic's facts. Meaningless on a "
        "fit request, which is why that queue carries topic_index IS NOT NULL. A "
        "fit replaces every row here, so a refit returns every topic to `new`; "
        "the questions themselves are not a topic's and survive it.",
    )
    question_error: Mapped[str | None] = mapped_column(
        Text, comment="Why question generation failed on this topic."
    )
    question_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When a worker claimed this topic for question generation, NULL "
        "when none holds it.",
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the fit that produced this topic was asked for.",
    )
    fitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="When the fit completed."
    )
    corpus_passages: Mapped[int | None] = mapped_column(
        Integer,
        comment="How many passages of this language the model was fitted over. "
        "Compared against the live count to tell whether the topics still "
        "describe the corpus.",
    )
    corpus_vocabulary: Mapped[int | None] = mapped_column(
        Integer, comment="How many terms survived the frequency filter."
    )
    passages_without_topics: Mapped[int | None] = mapped_column(
        Integer,
        comment="How many passages the fit could place in no topic. Those, and the "
        "facts drawn from them, are absent from every topic-weighted report.",
    )

    passage_links: Mapped[list[PassageTopic]] = relationship(
        back_populates="topic", cascade="all, delete-orphan", passive_deletes=True
    )
