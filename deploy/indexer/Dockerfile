# GraphRAG Indexer Job — Cloud Run Job container
# Build from repo root: docker build -f deploy/indexer/Dockerfile -t graphrag-indexer .

FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# ---- Copy workspace definition (for uv sync) ----
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Install all workspace packages (no dev dependencies)
# --all-packages is required to install workspace members (graphrag, graphrag-storage, etc.)
RUN uv sync --all-packages --no-dev --frozen

# ---- Indexer entrypoint ----
COPY deploy/indexer/entrypoint.py ./entrypoint.py
COPY deploy/config/ ./config/

ENV GRAPHRAG_CONFIG_DIR=/app/config
ENV GRAPHRAG_IS_UPDATE=false

# Use venv Python directly (avoids uv run re-syncing at startup)
CMD ["/app/.venv/bin/python", "entrypoint.py"]
