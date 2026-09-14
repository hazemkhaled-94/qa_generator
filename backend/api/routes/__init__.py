"""One module per service the API fronts.

`documents` is the ingestion service; `parsing`, `chunking` and `extraction`
are the pipeline stages, each built from `stage_router`; `topics` is the
topic modelling stage, which owns its routes because its trigger is a
request rather than the stage before it; `passages` and `facts` read back
what those stages produced; `questions` is both a stage and its output, and
so carries both in one router; `system` is the platform itself.
"""

from api.routes.chunking import router as chunking_router
from api.routes.documents import router as documents_router
from api.routes.extraction import router as extraction_router
from api.routes.facts import router as facts_router
from api.routes.parsing import router as parsing_router
from api.routes.passages import router as passages_router
from api.routes.questions import router as questions_router
from api.routes.system import router as system_router
from api.routes.topics import router as topics_router

#: Included by main in this order. Each module owns its own paths; a stage
#: mounts under a prefix of its own name.
ROUTERS = (
    system_router,
    documents_router,
    parsing_router,
    chunking_router,
    extraction_router,
    topics_router,
    passages_router,
    facts_router,
    questions_router,
)

__all__ = [
    "ROUTERS",
    "chunking_router",
    "documents_router",
    "extraction_router",
    "facts_router",
    "parsing_router",
    "passages_router",
    "questions_router",
    "system_router",
    "topics_router",
]
