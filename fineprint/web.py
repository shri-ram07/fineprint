"""HTTP layer: routing, request limits, security headers and error mapping. No business logic.

Run with: uv run --env-file .env uvicorn --factory fineprint.web:create_app
"""

import logging
import time
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
from .llm import AnthropicLLM, LLMClient, LLMError
from .schemas import AssistRequest, AssistResponse

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 10 * 1024 * 1024
# The app holds an API key and serves only this machine. Rejecting other Host headers stops a
# malicious page from reaching it through DNS rebinding.
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(llm: LLMClient | None = None) -> FastAPI:
    """Build the app. Without an injected client, credentials are checked here, at startup."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    llm = llm or AnthropicLLM.from_env()

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

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

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

    # A plain `def` so FastAPI runs it in a worker thread: the model call is synchronous and
    # can take minutes, and must not block the event loop.
    @app.post("/api/assist")
    def run_assistant(request: AssistRequest) -> AssistResponse:
        started = time.perf_counter()
        response = assist(request, llm)
        logger.info(
            "task=%s documents=%s warnings=%d duration=%.1fs",
            response.task,
            [len(document) for document in request.documents],
            len(response.warnings),
            time.perf_counter() - started,
        )
        return response

    return app
