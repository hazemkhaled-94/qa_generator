"""The passage_topics table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, Float, ForeignKey, Index, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base

if TYPE_CHECKING:
    from database.qa_generator.passages import Passage
    from database.qa_generator.topics import Topic


class PassageTopic(Base):
    """How strongly one passage belongs to one topic."""

    __tablename__ = "passage_topics"
    __table_args__ = (
        # The dominant-topic read walks a passage's memberships by weight.
        Index("ix_passage_topics_weight", "passage_id", "weight"),
        CheckConstraint(
            "weight > 0 AND weight <= 1", name="passage_topics_weight_is_a_probability"
        ),
        {
            "comment": "The topic distribution, one row per passage-topic pair above "
            "the weight floor. The only place the relationship is stored; the "
            "dominant topic is the highest weight, derived rather than stored."
        },
    )

    passage_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("passages.id", ondelete="CASCADE"),
        primary_key=True,
        comment="The passage. Re-chunking deletes it and these rows with it, "
        "which is what leaves a fit stale.",
    )
    topic_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("topics.id", ondelete="CASCADE"),
        primary_key=True,
        comment="The topic.",
    )
    weight: Mapped[float] = mapped_column(
        Float,
        comment="Share of this passage attributed to this topic, in (0, 1]. The "
        "weights for one passage sum to at most 1, which spans rows and so is not "
        "enforced; the only writer is the topic model.",
    )

    passage: Mapped[Passage] = relationship(back_populates="topic_links")
    topic: Mapped[Topic] = relationship(back_populates="passage_links")


#: Each passage's strongest topic, one row per passage. Declared beside the
#: table rather than in either service, because two of them read it and the
#: dominant topic is a property of this membership rather than of whoever
#: asks: topic modelling counts what each topic owns, question generation
#: takes the facts a topic is the subject of.
#:
#: DISTINCT ON rather than a max-weight join, so two topics tied at the same
#: weight yield one row instead of two.
#:
#: The topic id breaks the tie, and it is not decoration. Without it two
#: topics at the same weight leave WHICH of them is dominant to the plan,
#: and this subquery is read by four different queries in two services -
#: the facts a topic is the subject of, the passages a bridge is grouped
#: from, what a listing is filtered by and what a question's topic_scope is
#: counted over. Two of them disagreeing about one passage is a question
#: stored under a subject the page it was drawn on does not list it under,
#: and every other ordering in this pipeline already names its tiebreaker
#: for exactly this reason.
DOMINANT = (
    select(PassageTopic.passage_id, PassageTopic.topic_id)
    .order_by(
        PassageTopic.passage_id, PassageTopic.weight.desc(), PassageTopic.topic_id
    )
    .distinct(PassageTopic.passage_id)
    .subquery()
)
