# Production image for Cloud Run (or any container host).
FROM python:3.11-slim
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

# Cloud Run sets PORT and sits behind Google's front end, so trust its forwarding headers.
CMD exec uvicorn --factory fineprint.web:create_app --host 0.0.0.0 --port "${PORT:-8080}" \
    --proxy-headers --forwarded-allow-ips="*"
