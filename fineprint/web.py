"""HTTP layer: routing, request limits, security headers and error mapping. No business logic.

Run with: uv run --env-file .env uvicorn --factory fineprint.web:create_app
"""

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .assistant import assist
from .documents import DocumentError, DocumentKind, extract_text
from .llm import GeminiLLM, LLMClient, LLMError
from .schemas import AssistRequest, AssistResponse

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 10 * 1024 * 1024
# Rejecting unknown Host headers stops a malicious page from reaching a locally running copy
# through DNS rebinding. A deployment adds its own hostname via ALLOWED_HOSTS.
DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class RateLimiter:
    """Allow at most `limit` events per key in any rolling window.

    ponytail: in-process memory, so it only holds with a single instance (the deployment runs
    with --max-instances 1). Move to a shared store such as Redis if the service scales out.
    """

    MAX_KEYS = 10_000

    def __init__(self, limit: int, window_seconds: float = 3600) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if len(self._events) > self.MAX_KEYS:  # bound memory against many distinct clients
            self._events = defaultdict(
                deque, {k: v for k, v in self._events.items() if v and now - v[-1] < self.window}
            )
        events = self._events[key]
        while events and now - events[0] >= self.window:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


def create_app(llm: LLMClient | None = None) -> FastAPI:
    """Build the app. Without an injected client, credentials are checked here, at startup."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    llm = llm or GeminiLLM.from_env()
    allowed_hosts = os.environ.get("ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",")
    # Every /api/assist call spends the server's Gemini quota. The per-client limit keeps one
    # visitor from using it all; the total limit is the hard cap, since client addresses taken
    # from proxy headers can be spoofed.
    per_client = RateLimiter(int(os.environ.get("ASSIST_LIMIT_PER_CLIENT", "20")))
    in_total = RateLimiter(int(os.environ.get("ASSIST_LIMIT_TOTAL", "200")))

    # The interactive API docs load scripts from a CDN, which the CSP forbids; the README
    # documents the API endpoints instead.
    app = FastAPI(title="FinePrint", docs_url=None, redoc_url=None, openapi_url=None)

    # Middleware added later wraps middleware added earlier. The body limit rejects oversized
    # bodies (by header and by bytes received) before a route reads them; it must sit inside
    # the headers middleware, whose task group would otherwise swallow its 413.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_BODY_BYTES)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.exception_handler(DocumentError)
    async def document_error(_: Request, exc: DocumentError) -> JSONResponse:
        status = 413 if exc.code == "too_large" else 400
        return JSONResponse({"detail": exc.message}, status_code=status)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        status = 503 if exc.code == "unavailable" else 502
        return JSONResponse({"detail": exc.message}, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Built from field paths and messages only: pydantic's default error body echoes the
        # rejected input, which here would be the user's whole document.
        errors = exc.errors()
        logger.info("Rejected invalid request at %s", [error["loc"] for error in errors])
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
            for error in errors
        )
        return JSONResponse({"detail": detail}, status_code=422)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.post("/api/documents/text")
    async def document_text(request: Request, kind: DocumentKind) -> dict[str, str]:
        """Extract text from a file sent as the raw request body (`?kind=pdf|docx|txt|md`)."""
        data = await request.body()
        text = await run_in_threadpool(extract_text, kind, data)
        logger.info("Extracted %d characters from a %s upload", len(text), kind)
        return {"text": text}

    @app.post("/api/assist", response_model=AssistResponse)
    async def run_assistant(request: AssistRequest, http: Request) -> AssistResponse | JSONResponse:
        client = http.client.host if http.client else "unknown"
        if not (per_client.allow(client) and in_total.allow("*")):
            logger.warning("Rate limit reached")
            return JSONResponse(
                {"detail": "Too many requests right now. Please try again in a while."},
                status_code=429,
            )
        started = time.perf_counter()
        response = await assist(request, llm)
        logger.info(
            "task=%s documents=%s warnings=%d duration=%.1fs",
            response.task,
            [len(document) for document in request.documents],
            len(response.warnings),
            time.perf_counter() - started,
        )
        return response

    return app
