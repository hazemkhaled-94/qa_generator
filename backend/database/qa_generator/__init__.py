"""Tables of the qa_generator application database.

Importing this package registers every model on ``Base.metadata``, which is
what Alembic compares the live schema against.
"""

from database.qa_generator.base import Base
from database.qa_generator.documents import Document
from database.qa_generator.engine import engine, sessions
from database.qa_generator.fact_passages import FactPassage
from database.qa_generator.facts import Fact
from database.qa_generator.ingest_events import IngestEvent
from database.qa_generator.outcomes import (
    AnswerForm,
    Difficulty,
    DocumentScope,
    FactKind,
    Outcome,
    PassageScope,
    QuestionRejection,
    QuestionStatus,
    QuestionType,
    Rejection,
    ReviewVerdict,
    TopicScope,
    one_of,
)
from database.qa_generator.passage_topics import PassageTopic
from database.qa_generator.passages import Passage
from database.qa_generator.question_facts import QuestionFact
from database.qa_generator.questions import Question
from database.qa_generator.service_settings import ServiceSetting
from database.qa_generator.status import Status, check, queued
from database.qa_generator.topics import Topic

__all__ = [
    "AnswerForm",
    "Base",
    "Difficulty",
    "Document",
    "DocumentScope",
    "Fact",
    "FactKind",
    "FactPassage",
    "IngestEvent",
    "Outcome",
    "Passage",
    "PassageScope",
    "PassageTopic",
    "Question",
    "QuestionFact",
    "QuestionRejection",
    "QuestionStatus",
    "QuestionType",
    "Rejection",
    "ReviewVerdict",
    "ServiceSetting",
    "Status",
    "Topic",
    "TopicScope",
    "check",
    "engine",
    "one_of",
    "queued",
    "sessions",
]
