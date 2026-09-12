"""Routes about the platform itself rather than about any one service."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import status_service
from api.status import Component

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Reports that the process is up, for the container healthcheck.

    Touches no dependency; /status is the deep check.
    """
    return {"status": "ok"}


@router.get("/status")
def status() -> dict[str, Component]:
    """Reports the state of everything behind the API."""
    return status_service.snapshot()
