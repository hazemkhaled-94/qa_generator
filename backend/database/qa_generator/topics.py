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
    Integer,
    Text,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.status import Status, check

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
        {
            "comment": "Clusters over one language's own vocabulary, factorised out "
            "of a tf-idf weighted term matrix. "
            "passage_topics carries the membership; facts and questions reach "
            "their topics by joining through their passage. Also the queue: a row "
            "with a NULL topic_index is a request to refit."
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
        Text, comment="Name assigned by a person, if any."
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
