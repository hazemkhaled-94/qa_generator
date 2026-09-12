"""Tables of the qa_generator application database.

Importing this package registers every model on ``Base.metadata``, which is
what Alembic compares the live schema against.
"""

from database.qa_generator.base import Base
from database.qa_generator.documents import Document
from database.qa_generator.engine import engine, sessions
from database.qa_generator.facts import Fact
from database.qa_generator.ingest_events import IngestEvent
from database.qa_generator.outcomes import (
    Outcome,
    QuestionStatus,
    Rejection,
    one_of,
)
from database.qa_generator.passage_topics import PassageTopic
from database.qa_generator.passages import Passage
from database.qa_generator.question_facts import QuestionFact
from database.qa_generator.questions import Question
from database.qa_generator.status import Status, check, queued
from database.qa_generator.topics import Topic

__all__ = [
    "Base",
    "Document",
    "Fact",
    "IngestEvent",
    "Outcome",
    "Passage",
    "PassageTopic",
    "Question",
    "QuestionFact",
    "QuestionStatus",
    "Rejection",
    "Status",
    "Topic",
    "check",
    "engine",
    "one_of",
    "queued",
    "sessions",
]
