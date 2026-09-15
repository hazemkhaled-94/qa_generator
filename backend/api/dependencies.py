"""The wired instances every route shares.

The composition root, and the only module that knows which services exist.

Every import names a submodule rather than its package, so the API pays for
SQLAlchemy and nothing else: no Docling, no litellm, no gensim, no spaCy.
"""

from __future__ import annotations

import telemetry
from api.status import Counter, StatusService
from blob_store.seaweedfs import ExportBucket
from database.qa_generator import engine
from extraction.repository import FactCatalog, PassageQueue
from ingestion.config import Settings
from ingestion.factory import build_removal, build_service
from ingestion.repository import DocumentRepository
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from preprocessing.parsing.repository import ParseQueue
from question_generation.config import Settings as QuestionSettings
from question_generation.repository import QuestionCatalog, QuestionQueue
from topic_modelling.repository import TopicCatalog, TopicQueue

settings = Settings.load()

#: What generation was asked to write. Read here so a bad mix stops the API at
#: start-up naming itself, as every other setting does, and so the page can
#: put what was asked for beside what came out.
question_settings = QuestionSettings.load()

# Before anything else: a line logged earlier carries no trace id.
telemetry.configure("api")

# Builds the engine as a side effect, so an unreachable database fails this
# import rather than the first upload.
telemetry.trace_engine(engine())

#: One queue per stage, for the routes that move rows on and off it.
parsing_queue = ParseQueue()
chunking_queue = ChunkQueue()
extraction_queue = PassageQueue()
topics_queue = TopicQueue()
questions_queue = QuestionQueue()

#: One catalogue per thing the pipeline produced, for the routes that read it.
ingestion_repository = DocumentRepository()
passage_catalog = PassageCatalog()
fact_catalog = FactCatalog()
topic_catalog = TopicCatalog()
question_catalog = QuestionCatalog()

#: Generated artefacts a route serves back, such as the topic visualisations.
export_bucket = ExportBucket()

#: One instance per service the API fronts.
ingest_service = build_service(settings)
removal_service = build_removal()

#: What each service reports about itself on the status panel, which renders
#: whatever is here without knowing the names.
status_service = StatusService(
    {
        "ingestion": Counter(
            "Documents held and upload attempts made.", ingestion_repository.counts
        ),
        "parsing": Counter(
            "Documents by parse status.", parsing_queue.counts_by_status
        ),
        "chunking": Counter(
            "Documents by chunk status, and passages held.", chunking_queue.counts
        ),
        "extraction": Counter(
            "Passages by extract status, and facts held.", extraction_queue.counts
        ),
        "topic_modelling": Counter(
            "Topic runs by status, and topics held.", topics_queue.counts
        ),
        "question_generation": Counter(
            "Topics by question status, and questions held.", questions_queue.counts
        ),
    }
)
