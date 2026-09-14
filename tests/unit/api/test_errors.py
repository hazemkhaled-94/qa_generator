"""The shape every deliberate refusal from the API takes.

A caller branches on `code` rather than on English.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, ErrorBody, install


@pytest.fixture
def client() -> TestClient:
    """An application that raises one refusal and nothing else."""
    app = FastAPI()
    install(app)

    @app.get("/refuses")
    def refuses() -> None:
        """Raises the refusal under test."""
        raise ApiError(404, "unknown_document", "no document with that digest")

    @app.get("/answers")
    def answers() -> dict:
        """Answers normally."""
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_a_refusal_carries_its_status_code_and_detail() -> None:
    """Raised instead of HTTPException so the body carries a code."""
    error = ApiError(413, "too_large", "101.0 MB exceeds the 100 MB limit")

    assert error.status == 413
    assert error.body == ErrorBody(
        code="too_large", detail="101.0 MB exceeds the 100 MB limit"
    )
    assert str(error) == "101.0 MB exceeds the 100 MB limit"


def test_the_handler_answers_with_the_body_at_the_chosen_status(client) -> None:
    """Exactly two fields, both strings."""
    response = client.get("/refuses")

    assert response.status_code == 404
    assert response.json() == {
        "code": "unknown_document",
        "detail": "no document with that digest",
    }


def test_an_ordinary_answer_is_left_alone(client) -> None:
    """Installing the handler changes nothing else."""
    assert client.get("/answers").json() == {"ok": True}


def test_fastapis_own_validation_keeps_its_shape(client) -> None:
    """A malformed request is 422 in the standard shape, not an ErrorBody."""
    response = client.get("/nowhere")

    assert response.status_code == 404
    assert "code" not in response.json()
