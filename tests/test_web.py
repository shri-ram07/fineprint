import logging

import pytest
from fastapi.testclient import TestClient

from fineprint import web
from fineprint.documents import MAX_DOCUMENT_CHARS
from fineprint.llm import LLMClient, LLMError
from fineprint.web import SECURITY_HEADERS, RateLimiter, rate_limit_key


def make_client(llm: LLMClient, host: str = "localhost") -> TestClient:
    return TestClient(web.create_app(llm), base_url=f"http://{host}")


@pytest.fixture
def client(llm) -> TestClient:
    return make_client(llm)


def test_index_is_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FinePrint" in response.text
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-cache"


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
    client = make_client(llm)
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


def test_deployment_hostname_can_be_allowed(monkeypatch, llm):
    monkeypatch.setenv("ALLOWED_HOSTS", "fineprint.example")
    client = make_client(llm, host="fineprint.example")
    assert client.get("/").status_code == 200
    assert client.get("/", headers={"Host": "localhost"}).status_code == 400


def test_assist_is_rate_limited(monkeypatch, llm, lease):
    monkeypatch.setenv("ASSIST_LIMIT_PER_CLIENT", "1")
    client = make_client(llm)
    assert client.post("/api/assist", json={"documents": [lease]}).status_code == 200
    limited = client.post("/api/assist", json={"documents": [lease]})
    assert limited.status_code == 429
    assert "try again" in limited.json()["detail"]
    assert len(llm.calls) == 1  # the model was not called for the rejected request


def test_rate_limiter_uses_a_rolling_window():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("a", now=0) and limiter.allow("a", now=1)
    assert not limiter.allow("a", now=59)
    assert limiter.allow("b", now=59)  # other clients are unaffected
    assert limiter.allow("a", now=60.5)  # the first event has left the window


def test_static_assets_are_cached_and_compressed(client):
    response = client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert response.headers["Cache-Control"] == web.STATIC_CACHE_CONTROL


def test_spoofed_forwarding_headers_cannot_reset_the_limit(monkeypatch, llm, lease):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("ASSIST_LIMIT_PER_CLIENT", "1")
    client = make_client(llm)

    def post(forged: str):
        # The proxy appends the real address (203.0.113.7); the client controls the rest.
        headers = {"X-Forwarded-For": f"{forged}, 203.0.113.7"}
        return client.post("/api/assist", json={"documents": [lease]}, headers=headers)

    assert post("10.0.0.1").status_code == 200
    assert post("10.0.0.2").status_code == 429


def test_uploads_are_rate_limited(monkeypatch, llm):
    monkeypatch.setenv("UPLOAD_LIMIT_PER_CLIENT", "1")
    client = make_client(llm)
    assert client.post("/api/documents/text?kind=txt", content=b"Rent").status_code == 200
    assert client.post("/api/documents/text?kind=txt", content=b"Rent").status_code == 429


@pytest.mark.parametrize(
    ("origin", "status"),
    [("https://evil.example", 403), ("null", 403), ("http://localhost", 200), (None, 200)],
    ids=["foreign-site", "opaque-origin", "same-origin", "no-origin"],
)
def test_cross_site_posts_are_refused(client, origin, status):
    headers = {"Origin": origin} if origin else {}
    response = client.post("/api/documents/text?kind=txt", content=b"Rent", headers=headers)
    assert response.status_code == status


def test_identical_requests_are_answered_from_the_cache(client, llm, lease):
    first = client.post("/api/assist", json={"documents": [lease]})
    second = client.post("/api/assist", json={"documents": [lease]})
    assert first.json() == second.json()
    assert len(llm.calls) == 1


def test_total_cap_counts_model_calls_across_clients(monkeypatch, llm, lease):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("ASSIST_LIMIT_TOTAL", "1")
    client = make_client(llm)

    def post(address: str, context: str = ""):
        headers = {"X-Forwarded-For": address}
        body = {"documents": [lease], "context": context}
        return client.post("/api/assist", json=body, headers=headers)

    assert post("203.0.113.7").status_code == 200
    assert post("198.51.100.9").status_code == 200  # answered from the cache: no model call
    assert post("198.51.100.9", context="I'm the landlord").status_code == 429
    assert len(llm.calls) == 1


def test_missing_forwarding_header_falls_back_to_the_socket_address(monkeypatch, llm):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("UPLOAD_LIMIT_PER_CLIENT", "1")
    client = make_client(llm)
    assert client.post("/api/documents/text?kind=txt", content=b"Rent").status_code == 200
    assert client.post("/api/documents/text?kind=txt", content=b"Rent").status_code == 429


@pytest.mark.parametrize(
    ("address", "key"),
    [
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:bbbb::9", "2001:db8:1:2::/64"),
        ("203.0.113.7", "203.0.113.7"),
        ("testclient", "testclient"),
    ],
)
def test_ipv6_clients_are_limited_per_network(address, key):
    assert rate_limit_key(address) == key


def test_rate_limiter_forgets_the_least_recently_seen_client(monkeypatch):
    monkeypatch.setattr(RateLimiter, "MAX_KEYS", 2)
    limiter = RateLimiter(limit=1)
    assert limiter.allow("a", now=0)
    assert limiter.allow("b", now=1)
    assert not limiter.allow("a", now=2)  # "a" is seen again, so "b" is now the oldest
    assert limiter.allow("c", now=3)  # evicts "b", not "a"
    assert not limiter.allow("a", now=4)  # "a" kept its history
    assert limiter.allow("b", now=5)  # "b" starts afresh
