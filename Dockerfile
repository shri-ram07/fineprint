# Production image for Cloud Run (or any container host).
# The base image is pinned by digest for reproducible, tamper-evident builds.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY fineprint ./fineprint
RUN uv sync --locked

RUN useradd --no-create-home app
USER app

# Cloud Run sets PORT. Client addresses for rate limiting are read by the app itself
# (TRUSTED_PROXY_HOPS), not by uvicorn's proxy mode, which trusts the forgeable leftmost hop.
CMD exec uvicorn --factory fineprint.web:create_app --host 0.0.0.0 --port "${PORT:-8080}"
