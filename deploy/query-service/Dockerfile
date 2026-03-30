# GraphRAG Query Service — Cloud Run container
# Build from repo root: docker build -f deploy/query-service/Dockerfile -t graphrag-query .

FROM python:3.11-slim AS base

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# ---- Copy workspace definition (for uv sync) ----
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Install all workspace packages (no dev dependencies)
# --all-packages is required to install workspace members (graphrag, graphrag-storage, etc.)
RUN uv sync --all-packages --no-dev --frozen

# Install FastAPI/uvicorn (not in workspace deps — needed for the HTTP server only)
COPY deploy/query-service/requirements.txt ./query-requirements.txt
RUN uv pip install --no-cache-dir -r query-requirements.txt

# ---- App code ----
COPY deploy/query-service/app/ ./app/
COPY deploy/config/ ./config/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
ENV GRAPHRAG_CONFIG_DIR=/app/config

EXPOSE 8080

# Use venv Python directly (avoids uv run re-syncing at startup)
# Single worker — SpannerResourceManager uses module-level state not safe across processes
# Note: --loop uvloop is intentionally omitted — nest_asyncio2 (used internally
# by graphrag) is incompatible with uvloop and raises ValueError at startup.
CMD ["/app/.venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1"]
