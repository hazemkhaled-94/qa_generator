"""One module per service the API fronts.

`documents` is the ingestion service; `parsing`, `chunking` and `extraction`
are the pipeline stages, each built from `stage_router`; `topics` is the
topic modelling stage, which owns its routes because its trigger is a
request rather than the stage before it; `passages` and `facts` read back
what those stages produced; `questions` is both a stage and its output, and
so carries both in one router; `assessment` is the evaluation phase, which
judges what those stages produced and carries both in one router for the
same reason; `settings` is what each of them is configured to do, one
service per request; `prompts` is what a version of one asked for; `lineage`
is how one artefact any of them produced was produced; `system` is the
platform itself.
"""

from api.routes.assessment import router as assessment_router
from api.routes.chunking import router as chunking_router
from api.routes.documents import router as documents_router
from api.routes.extraction import router as extraction_router
from api.routes.facts import router as facts_router
from api.routes.lineage import router as lineage_router
from api.routes.parsing import router as parsing_router
from api.routes.passages import router as passages_router
from api.routes.prompts import router as prompts_router
from api.routes.questions import router as questions_router
from api.routes.settings import router as settings_router
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
    assessment_router,
    settings_router,
    prompts_router,
    lineage_router,
)

__all__ = [
    "ROUTERS",
    "assessment_router",
    "chunking_router",
    "documents_router",
    "extraction_router",
    "facts_router",
    "lineage_router",
    "parsing_router",
    "passages_router",
    "prompts_router",
    "questions_router",
    "settings_router",
    "system_router",
    "topics_router",
]
