import logging

import pytest
from fastapi.testclient import TestClient

from fineprint import web
from fineprint.documents import MAX_DOCUMENT_CHARS
from fineprint.llm import LLMError


@pytest.fixture
def client(llm) -> TestClient:
    return TestClient(web.create_app(llm), base_url="http://localhost")


def test_index_is_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FinePrint" in response.text
    csp = response.headers["Content-Security-Policy"]
    assert csp == "default-src 'self'; frame-ancestors 'none'"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200


def test_api_docs_are_disabled(client):
    assert client.get("/docs").status_code == 404


def test_extracts_text_from_a_raw_upload(client):
    response = client.post("/api/documents/text?kind=txt", content=b"Rent is due monthly.")
    assert response.json() == {"text": "Rent is due monthly."}


def test_unsupported_file_kind_is_rejected_readably(client):
    response = client.post("/api/documents/text?kind=exe", content=b"MZ")
    assert response.status_code == 422
    assert response.json()["detail"].startswith("kind:")


def test_unreadable_file_is_a_client_error(client):
    response = client.post("/api/documents/text?kind=pdf", content=b"%PDF-1.7 garbage")
    assert response.status_code == 400
    assert "could not be read" in response.json()["detail"]


def test_oversized_body_is_rejected_before_parsing(monkeypatch, llm):
    monkeypatch.setattr(web, "MAX_BODY_BYTES", 64)
    client = TestClient(web.create_app(llm), base_url="http://localhost")
    response = client.post("/api/documents/text?kind=txt", content=b"x" * 65)
    assert response.status_code == 413


def test_foreign_host_is_rejected(client):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_assist_returns_the_verified_result(client, lease):
    response = client.post("/api/assist", json={"documents": [lease], "context": "I'm the tenant"})
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "analyze"
    assert body["result"]["risks"][0]["severity"] == "high"
    assert body["warnings"] == []
    assert "not legal advice" in body["disclaimer"]


def test_validation_errors_never_echo_the_document(client, caplog):
    caplog.set_level(logging.DEBUG)
    document = "CONFIDENTIAL-SENTINEL " + "x" * MAX_DOCUMENT_CHARS
    response = client.post("/api/assist", json={"documents": [document]})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("documents.0:")
    assert "CONFIDENTIAL-SENTINEL" not in response.text
    assert "CONFIDENTIAL-SENTINEL" not in caplog.text


@pytest.mark.parametrize(
    ("code", "status"),
    [("unavailable", 503), ("truncated", 502), ("configuration", 502)],
)
def test_model_failures_return_fixed_messages(client, llm, lease, code, status):
    llm.error = LLMError(code)
    response = client.post("/api/assist", json={"documents": [lease]})
    assert response.status_code == status
    assert response.json() == {"detail": LLMError(code).message}
