"""Shared pytest fixtures.

Every test forces ``OFFLINE=1`` so nothing is downloaded and the deterministic
stub backs the API. The ``client`` fixture drives the real ASGI app through a
``TestClient``, which runs the lifespan (model load + batcher startup) exactly
as production would.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("OFFLINE", "1")
    app = create_app()
    with TestClient(app) as c:  # `with` triggers startup/shutdown lifespan
        yield c
