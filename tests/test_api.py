"""
Day 4 — API tests for /ask.

Uses FastAPI's TestClient to verify:
  - /health works
  - /ask returns a well-formed response for a simple question
"""

from fastapi.testclient import TestClient
from service.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_ask_basic():
    payload = {"question": "What is access control?", "top_k": 3}
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # Basic shape checks
    assert "question" in data
    assert "answer" in data
    assert "citations" in data
    assert "guardrails" in data

    assert isinstance(data["citations"], list)
    assert isinstance(data["guardrails"], dict)
    assert isinstance(data["answer"], str)
    assert data["answer"].strip() != ""