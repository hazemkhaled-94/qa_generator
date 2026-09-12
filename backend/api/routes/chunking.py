"""Queue routes for the chunking stage.

Names the queue and nothing else: the routes are in stage.py and the work
is in the worker, so no service is imported here.
"""

from __future__ import annotations

from api.dependencies import chunking_queue
from api.routes.stage import stage_router

router = stage_router(name="chunking", repository=chunking_queue)
