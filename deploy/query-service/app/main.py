# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""FastAPI query service wrapping the GraphRAG v3 search API.

Endpoints:
  POST /v1/query/global   — Global search (community reports, no vector store)
  POST /v1/query/local    — Local search (entities + vector similarity)
  POST /v1/query/drift    — DRIFT search (progressive refinement)
  POST /v1/query/basic    — Basic RAG search (text units only)
  GET  /healthz           — Liveness probe (always 200 once app starts)
  GET  /readyz            — Readiness probe (200 after index loaded, 503 during startup)
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import graphrag.api as api
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.loader import IndexData, load_index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

CONFIG_DIR = Path(os.getenv("GRAPHRAG_CONFIG_DIR", "/app/config"))

# ---------------------------------------------------------------------------
# Lifespan: load index once at startup, shared across all requests
# ---------------------------------------------------------------------------
_index: IndexData | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    logger.info("Loading index from GCS...")
    _index = await load_index(CONFIG_DIR)
    logger.info("Index loaded — service ready")
    yield
    _index = None


app = FastAPI(title="GraphRAG Query Service", version="3.0.0", lifespan=lifespan)


def _require_index() -> IndexData:
    if _index is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Index not loaded yet")
    return _index


def _serialize_context(context: Any) -> Any:
    """Convert context DataFrames to JSON-serializable dicts."""
    if isinstance(context, dict):
        return {k: _serialize_context(v) for k, v in context.items()}
    if hasattr(context, "to_dict"):
        return context.to_dict(orient="records")
    if isinstance(context, list):
        return [_serialize_context(item) for item in context]
    return context


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GlobalQueryRequest(BaseModel):
    query: str
    community_level: int | None = 2
    dynamic_community_selection: bool = False
    response_type: str = "Multiple Paragraphs"


class LocalQueryRequest(BaseModel):
    query: str
    community_level: int = 2
    response_type: str = "Multiple Paragraphs"


class DriftQueryRequest(BaseModel):
    query: str
    community_level: int = 2
    response_type: str = "Multiple Paragraphs"


class BasicQueryRequest(BaseModel):
    query: str
    response_type: str = "Multiple Paragraphs"


class QueryResponse(BaseModel):
    response: str | dict | list
    context_data: Any = None


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------
@app.get("/healthz", status_code=200)
async def healthz():
    return {"status": "ok"}


@app.get("/readyz", status_code=200)
async def readyz():
    if _index is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
        "entities": len(_index.entities),
        "communities": len(_index.communities),
        "community_reports": len(_index.community_reports),
        "text_units": len(_index.text_units),
    }


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/query/global", response_model=QueryResponse)
async def query_global(req: GlobalQueryRequest):
    """Global search: uses community reports, no per-entity vector lookup."""
    idx = _require_index()
    response, context = await api.global_search(
        config=idx.config,
        entities=idx.entities,
        communities=idx.communities,
        community_reports=idx.community_reports,
        community_level=req.community_level,
        dynamic_community_selection=req.dynamic_community_selection,
        response_type=req.response_type,
        query=req.query,
    )
    return QueryResponse(response=response, context_data=_serialize_context(context))


@app.post("/v1/query/local", response_model=QueryResponse)
async def query_local(req: LocalQueryRequest):
    """Local search: entity-level vector similarity + community context."""
    idx = _require_index()
    response, context = await api.local_search(
        config=idx.config,
        entities=idx.entities,
        communities=idx.communities,
        community_reports=idx.community_reports,
        text_units=idx.text_units,
        relationships=idx.relationships,
        covariates=idx.covariates,
        community_level=req.community_level,
        response_type=req.response_type,
        query=req.query,
    )
    return QueryResponse(response=response, context_data=_serialize_context(context))


@app.post("/v1/query/drift", response_model=QueryResponse)
async def query_drift(req: DriftQueryRequest):
    """DRIFT search: progressive community-level refinement."""
    idx = _require_index()
    response, context = await api.drift_search(
        config=idx.config,
        entities=idx.entities,
        communities=idx.communities,
        community_reports=idx.community_reports,
        text_units=idx.text_units,
        relationships=idx.relationships,
        community_level=req.community_level,
        response_type=req.response_type,
        query=req.query,
    )
    return QueryResponse(response=response, context_data=_serialize_context(context))


@app.post("/v1/query/basic", response_model=QueryResponse)
async def query_basic(req: BasicQueryRequest):
    """Basic RAG search: direct text-unit retrieval without graph traversal."""
    idx = _require_index()
    response, context = await api.basic_search(
        config=idx.config,
        text_units=idx.text_units,
        response_type=req.response_type,
        query=req.query,
    )
    return QueryResponse(response=response, context_data=_serialize_context(context))
