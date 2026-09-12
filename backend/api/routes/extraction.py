"""Queue routes for the extraction stage.

Names the queue and nothing else: the routes are in stage.py and the work
is in the worker, so no service is imported here.
"""

from __future__ import annotations

from api.dependencies import extraction_queue
from api.routes.stage import stage_router

router = stage_router(name="extraction", repository=extraction_queue)
