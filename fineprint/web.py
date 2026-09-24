"""HTTP layer: routing, request limits, security headers and error mapping. No business logic.

Run with: uv run --env-file .env uvicorn --factory fineprint.web:create_app
"""

import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .assistant import ResultCache, assist
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
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; object-src 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "Strict-Transport-Security": "max-age=31536000",  # ignored by browsers over plain http
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
STATIC_CACHE_CONTROL = "public, max-age=3600"
TOO_MANY_REQUESTS = "Too many requests right now. Please try again in a while."


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


def client_address(request: Request, trusted_proxy_hops: int) -> str:
    """The caller's IP address, used as the rate-limit key.

    Behind `trusted_proxy_hops` proxies (1 on Cloud Run), X-Forwarded-For is read from the
    right: each proxy appends the address it saw, while everything further left was written
    by the client and can be forged. With no trusted proxy, the socket peer is used.
    """
    if trusted_proxy_hops:
        hops = [hop.strip() for hop in request.headers.get("x-forwarded-for", "").split(",")]
        hops = [hop for hop in hops if hop]
        if len(hops) >= trusted_proxy_hops:
            return hops[-trusted_proxy_hops]
    return request.client.host if request.client else "unknown"


def _limit(proxy_hops: int, per_client: RateLimiter, total: RateLimiter | None = None) -> Callable:
    """A route dependency that answers 429 once a client, or everyone together, hits a limit."""

    def check(request: Request) -> None:
        within = per_client.allow(client_address(request, proxy_hops))
        if not within or (total is not None and not total.allow("*")):
            logger.warning("Rate limit reached on %s", request.url.path)
            raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS)

    return check


def _setting(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def create_app(llm: LLMClient | None = None) -> FastAPI:
    """Build the app. Without an injected client, credentials are checked here, at startup."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    llm = llm or GeminiLLM.from_env()
    allowed_hosts = os.environ.get("ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",")
    proxy_hops = _setting("TRUSTED_PROXY_HOPS", 0)
    cache = ResultCache(_setting("RESULT_CACHE_SIZE", 128))
    # Every /api/assist call spends the server's Gemini quota: the per-client limit keeps one
    # visitor from using it all, and the total limit is a hard cap on spend. Uploads cost only
    # CPU, so they get a per-client limit alone.
    limit_assist = _limit(
        proxy_hops,
        RateLimiter(_setting("ASSIST_LIMIT_PER_CLIENT", 20)),
        RateLimiter(_setting("ASSIST_LIMIT_TOTAL", 200)),
    )
    limit_upload = _limit(proxy_hops, RateLimiter(_setting("UPLOAD_LIMIT_PER_CLIENT", 60)))

    # The interactive API docs load scripts from a CDN, which the CSP forbids; the README
    # documents the API endpoints instead.
    app = FastAPI(title="FinePrint", docs_url=None, redoc_url=None, openapi_url=None)

    # Middleware added later wraps middleware added earlier. The body limit rejects oversized
    # bodies (by header and by bytes received) before a route reads them; it must sit inside
    # the headers middleware, whose task group would otherwise swallow its 413.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_BODY_BYTES)

    @app.middleware("http")
    async def guard_and_harden(request: Request, call_next):
        # Browsers send Origin on cross-site POSTs. The upload endpoint takes a raw body, which
        # a foreign page could post without a CORS preflight, so cross-site POSTs are refused.
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and urlsplit(origin).netloc != request.url.netloc:
            response = JSONResponse({"detail": "Cross-site requests are not allowed."}, 403)
        else:
            response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = STATIC_CACHE_CONTROL
        return response

    app.add_middleware(GZipMiddleware, minimum_size=1000)
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
        # Always revalidate the page itself so a new deployment is picked up immediately.
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.post("/api/documents/text", dependencies=[Depends(limit_upload)])
    async def document_text(request: Request, kind: DocumentKind) -> dict[str, str]:
        """Extract text from a file sent as the raw request body (`?kind=pdf|docx|txt|md`)."""
        data = await request.body()
        text = await run_in_threadpool(extract_text, kind, data)
        logger.info("Extracted %d characters from a %s upload", len(text), kind)
        return {"text": text}

    @app.post("/api/assist", dependencies=[Depends(limit_assist)])
    async def run_assistant(request: AssistRequest) -> AssistResponse:
        started = time.perf_counter()
        response = await assist(request, llm, cache)
        logger.info(
            "task=%s documents=%s warnings=%d duration=%.1fs",
            response.task,
            [len(document) for document in request.documents],
            len(response.warnings),
            time.perf_counter() - started,
        )
        return response

    return app
