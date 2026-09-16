"""The question_facts table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base

if TYPE_CHECKING:
    from database.qa_generator.facts import Fact
    from database.qa_generator.questions import Question


class QuestionFact(Base):
    """Which facts a question was generated from.

    One row for an ordinary question, two or more for a cross-document one.
    The passages a question needs retrieved are reached from here through
    fact_passages rather than stored.
    """

    __tablename__ = "question_facts"
    __table_args__ = (
        {
            "comment": "Which facts a question was generated from. One row for an "
            "ordinary question, two or more for a cross-document question. The "
            "passages a question needs retrieved are derived from here through "
            "fact_passages rather than stored separately."
        },
    )

    question_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("questions.id", ondelete="CASCADE"),
        primary_key=True,
        comment="The question.",
    )
    fact_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("facts.id", ondelete="CASCADE"),
        primary_key=True,
        # The primary key indexes (question_id, fact_id), which is no help
        # to the lookup by fact_id alone that the cascade runs.
        index=True,
        comment="A fact the question tests. Deleting the fact removes this row; "
        "see the note on the questions table about questions left with no facts.",
    )

    question: Mapped[Question] = relationship(back_populates="fact_links")
    fact: Mapped[Fact] = relationship(back_populates="question_links")


#: Deletes a question once its last source fact is gone. A foreign key
#: cascades parent to child, never the reverse. Re-entrant safely: the second
#: firing finds the question already deleted and the NOT EXISTS guard stops
#: it.
#:
#: Declared here, beside the table it belongs to, and executed by the
#: migration that creates that table: autogenerate sees tables and columns,
#: never a trigger, so a migration has to name it.
DELETE_ORPHAN_FUNCTION = """
CREATE OR REPLACE FUNCTION delete_question_without_facts() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM questions q
     WHERE q.id = OLD.question_id
       AND NOT EXISTS (
           SELECT 1 FROM question_facts qf WHERE qf.question_id = q.id
       );
    RETURN NULL;
END;
$$;
"""

DELETE_ORPHAN_TRIGGER = """
CREATE TRIGGER question_facts_delete_orphan_question
AFTER DELETE ON question_facts
FOR EACH ROW EXECUTE FUNCTION delete_question_without_facts();
"""

DROP_DELETE_ORPHAN = """
DROP TRIGGER IF EXISTS question_facts_delete_orphan_question ON question_facts;
DROP FUNCTION IF EXISTS delete_question_without_facts();
"""
