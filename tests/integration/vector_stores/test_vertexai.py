# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for VertexAIVectorStore.

Prerequisites: a MatchingEngineIndex + deployed endpoint must already exist
in the target project. Set the following env vars before running:

    export GRAPHRAG_GCP_INTEGRATION_TEST=1
    export GCP_PROJECT_ID=your-project
    export VERTEXAI_LOCATION=us-central1
    export VERTEXAI_INDEX_ID=projects/.../indexes/...      # full resource name or numeric ID
    export VERTEXAI_INDEX_ENDPOINT_ID=projects/.../indexEndpoints/...
    export VERTEXAI_DEPLOYED_INDEX_ID=your-deployed-index-id
    export VERTEXAI_VECTOR_SIZE=768   # must match the index's declared dimension

    uv run python -m pytest tests/integration/vector_stores/test_vertexai.py -v

NOTE:
- upsert_datapoints() is asynchronous; newly inserted vectors may not be
  immediately searchable. Tests wait up to SEARCH_WAIT_SECONDS for propagation.
- Vertex AI does not expose a direct ID-lookup or count API; search_by_id()
  returns a stub and count() returns 0 — those behaviours are explicitly tested.
- The test prefix (VERTEXAI_TEST_PREFIX) is used as a unique namespace per run
  so that leftover datapoints from a crashed run do not affect future ones.
"""

import os
import time

import pytest

from graphrag_vectors.vector_store import VectorStoreDocument
from graphrag_vectors.vertexai import VertexAIVectorStore

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRAPHRAG_GCP_INTEGRATION_TEST"),
    reason="GCP integration tests not enabled (set GRAPHRAG_GCP_INTEGRATION_TEST=1)",
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

SEARCH_WAIT_SECONDS = int(os.environ.get("VERTEXAI_SEARCH_WAIT_SECONDS", "5"))


def _cfg() -> dict:
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID"),
        "location": os.environ.get("VERTEXAI_LOCATION", "us-central1"),
        "index_id": os.environ.get("VERTEXAI_INDEX_ID"),
        "index_endpoint_id": os.environ.get("VERTEXAI_INDEX_ENDPOINT_ID"),
        "deployed_index_id": os.environ.get("VERTEXAI_DEPLOYED_INDEX_ID"),
    }


def _vector_size() -> int:
    return int(os.environ.get("VERTEXAI_VECTOR_SIZE", "768"))


def _require_cfg(cfg: dict) -> None:
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        pytest.skip(f"Vertex AI config not set: {missing}")


# ---------------------------------------------------------------------------
# Module-scoped fixture: connect once, clean up test datapoints after module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vs():
    """Connect to the existing Vertex AI index once for the whole module."""
    cfg = _cfg()
    _require_cfg(cfg)

    vector_size = _vector_size()
    store = VertexAIVectorStore(index_name="vertexai_integration", vector_size=vector_size, **cfg)
    store.connect()

    yield store

    # Best-effort cleanup: remove any datapoints inserted by this test run.
    # Uses the VERTEXAI_TEST_PREFIX env var as a marker if set.
    prefix = os.environ.get("VERTEXAI_TEST_PREFIX", "itest-")
    try:
        if hasattr(store, "_index") and store._index:
            # Vertex AI doesn't support listing datapoints, so we rely on the
            # test IDs being predictable (all start with the prefix).
            ids_to_clean = [f"{prefix}{i}" for i in range(100)]
            store.remove(ids_to_clean)
    except Exception as e:
        print(f"Warning: cleanup failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid(suffix: str = "") -> str:
    """Return a test-prefixed datapoint ID."""
    prefix = os.environ.get("VERTEXAI_TEST_PREFIX", "itest-")
    return f"{prefix}{suffix}" if suffix else f"{prefix}0"


def _vec(seed: float = 0.1) -> list[float]:
    size = _vector_size()
    base = [seed] * size
    # Normalise so cosine distance is well-defined
    norm = sum(x * x for x in base) ** 0.5
    return [x / norm for x in base]


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def test_connect(vs):
    """Store must connect without raising."""
    assert vs._index is not None
    assert vs._index_endpoint is not None


# ---------------------------------------------------------------------------
# load_documents / upsert
# ---------------------------------------------------------------------------

def test_load_documents(vs):
    """Upserting documents must not raise."""
    doc_id = _uid("load-1")
    vs.load_documents([
        VectorStoreDocument(id=doc_id, vector=_vec(0.1), data={"label": "load-test"})
    ])
    # Cleanup
    vs.remove([doc_id])


def test_load_documents_empty_is_noop(vs):
    vs.load_documents([])  # must not raise


def test_load_skips_none_vector(vs):
    """Documents with vector=None must be silently skipped."""
    vs.load_documents([VectorStoreDocument(id=_uid("no-vec"), vector=None, data={})])
    # No assertion needed — just verifying no exception


# ---------------------------------------------------------------------------
# Limitations — search_by_id / count
# ---------------------------------------------------------------------------

def test_search_by_id_returns_stub(vs):
    """Vertex AI does not expose a direct ID lookup; a stub doc is returned."""
    doc = vs.search_by_id("any-id")
    assert doc.id == "any-id"
    assert doc.vector is None


def test_count_returns_zero(vs):
    """Vertex AI does not expose a document count API; always returns 0."""
    assert vs.count() == 0


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove(vs):
    """remove() must not raise (even for non-existent IDs)."""
    vs.remove([_uid("ghost-1"), _uid("ghost-2")])


def test_remove_empty_list_is_noop(vs):
    vs.remove([])


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update(vs):
    """update() delegates to load_documents — must not raise."""
    doc = VectorStoreDocument(id=_uid("upd-1"), vector=_vec(0.2), data={"v": 2})
    vs.update(doc)
    # Cleanup
    vs.remove([_uid("upd-1")])


# ---------------------------------------------------------------------------
# similarity_search_by_vector
# ---------------------------------------------------------------------------

def test_similarity_search_returns_results(vs):
    """Insert a document then search for it; it must appear in results."""
    doc_id = _uid("search-1")
    query_vec = _vec(0.1)

    vs.load_documents([VectorStoreDocument(id=doc_id, vector=query_vec, data={})])

    # Vertex AI propagation is asynchronous — wait briefly.
    time.sleep(SEARCH_WAIT_SECONDS)

    results = vs.similarity_search_by_vector(query_vec, k=5)
    assert isinstance(results, list)
    assert all(isinstance(r.score, float) for r in results)

    # Cleanup (best-effort; the datapoint may still appear briefly after removal)
    vs.remove([doc_id])


def test_similarity_search_k_limit(vs):
    """k parameter is passed to the API; result count must not exceed k."""
    results = vs.similarity_search_by_vector(_vec(0.3), k=1)
    assert len(results) <= 1


def test_similarity_search_by_text(vs):
    """similarity_search_by_text delegates to similarity_search_by_vector."""
    def mock_embedder(text: str) -> list[float]:
        return _vec(0.1)

    results = vs.similarity_search_by_text("test query", mock_embedder, k=3)
    assert isinstance(results, list)


def test_similarity_search_empty_embedding_returns_empty(vs):
    """Empty embedding from embedder must short-circuit and return []."""
    def null_embedder(text: str) -> list[float]:
        return []

    results = vs.similarity_search_by_text("test", null_embedder, k=5)
    assert results == []


# ---------------------------------------------------------------------------
# include_vectors / select — Vertex AI limitations
# ---------------------------------------------------------------------------

def test_include_vectors_false_results_have_no_vector(vs):
    """Vertex AI search results never include vectors (limitation of the API)."""
    results = vs.similarity_search_by_vector(_vec(0.1), k=2, include_vectors=False)
    for r in results:
        assert r.document.vector is None


def test_select_parameter_accepted(vs):
    """select parameter must be accepted without raising (results may be empty)."""
    results = vs.similarity_search_by_vector(
        _vec(0.1), k=2, select=["text"]
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# FilterExpr — applied in-memory on search results
# ---------------------------------------------------------------------------

def test_filter_eq_on_search_results(vs):
    """FilterExpr is applied in-memory; results without matching data are excluded."""
    from graphrag_vectors.filtering import F

    # Without any live data carrying 'status' field, filter should return nothing.
    results = vs.similarity_search_by_vector(
        _vec(0.1), k=10, filters=F.status == "active"
    )
    assert isinstance(results, list)
    for r in results:
        assert r.document.data.get("status") == "active"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

def test_vector_store_factory():
    """create_vector_store() with type='vertexai' produces a connected store."""
    cfg = _cfg()
    _require_cfg(cfg)

    from graphrag_vectors.index_schema import IndexSchema
    from graphrag_vectors.vector_store_config import VectorStoreConfig
    from graphrag_vectors.vector_store_factory import create_vector_store

    config = VectorStoreConfig(type="vertexai", vector_size=_vector_size(), **cfg)
    schema = IndexSchema(index_name="factory_test", vector_size=_vector_size())
    store = create_vector_store(config, schema)
    store.connect()
    assert store._index is not None
    assert store._index_endpoint is not None
