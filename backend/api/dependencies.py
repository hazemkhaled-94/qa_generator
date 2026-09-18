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
from ingestion.service import IngestService
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from preprocessing.parsing.repository import ParseQueue
from question_generation.catalog import QuestionCatalog
from question_generation.config import Settings as QuestionSettings
from question_generation.queue import QuestionQueue
from settings.store import resolved
from topic_modelling.repository import TopicCatalog, TopicQueue

#: Read at import so a bad allowlist stops the API at start-up naming itself,
#: as every other setting does, rather than on the first upload.
settings = Settings.load()

#: The same, for what generation was asked to write: a mix naming a type
#: nothing answers to is a start-up failure and not a page that will not draw.
#:
#: Neither is what a route answers with. A stored setting can change while the
#: API runs, so anything that reports or enforces one reads it per request -
#: `ingesting` below, and /questions/plan. These two exist to fail early.
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


def ingesting() -> IngestService:
    """The ingest service, built from the settings as they stand.

    Ingestion is the one service with no worker: the API is what runs it, so
    a setting it reads has nowhere else to be picked up. The upload route
    builds it per request for that reason, and the instance above serves the
    routes that only read what is already stored.

    Cheap to build. The repository captures the cached session factory and
    the bucket wraps the cached S3 client, so this is three settings and two
    thin objects rather than a connection.
    """
    return build_service(Settings.load(resolved()))


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
