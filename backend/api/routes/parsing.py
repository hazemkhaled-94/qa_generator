"""Queue routes for the parsing stage.

Names the queue and nothing else: the routes are in stage.py and the work
is in the worker, so no service is imported here.
"""

from __future__ import annotations

from api.dependencies import parsing_queue
from api.routes.stage import stage_router

router = stage_router(name="parsing", repository=parsing_queue)
