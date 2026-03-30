# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for SpannerVectorStore — mirrors test_lancedb.py coverage.

Run with real GCP credentials:

    export GRAPHRAG_GCP_INTEGRATION_TEST=1
    export GCP_PROJECT_ID=your-project
    export SPANNER_INSTANCE_ID=your-instance
    export SPANNER_DATABASE_ID=your-database
    uv run python -m pytest tests/integration/vector_stores/test_spanner.py -v

NOTE: Spanner vector index DDL can take 5–15 minutes to build.
      A module-scoped fixture creates the table/index ONCE and truncates rows
      between individual tests to avoid per-test DDL overhead.
"""

import os
from uuid import uuid4

import pytest
from google.cloud import spanner as spanner_lib

from graphrag_vectors.filtering import F
from graphrag_vectors.spanner import SpannerVectorStore
from graphrag_vectors.vector_store import VectorStoreDocument

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRAPHRAG_GCP_INTEGRATION_TEST"),
    reason="GCP integration tests not enabled (set GRAPHRAG_GCP_INTEGRATION_TEST=1)",
)

VECTOR_SIZE = 5


def _spanner_config() -> dict:
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID"),
        "instance_id": os.environ.get("SPANNER_INSTANCE_ID"),
        "database_id": os.environ.get("SPANNER_DATABASE_ID"),
    }


def _db(cfg: dict):
    client = spanner_lib.Client(project=cfg["project_id"])
    return client.instance(cfg["instance_id"]).database(cfg["database_id"])


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
    VectorStoreDocument(id="2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
    VectorStoreDocument(id="3", vector=[0.3, 0.4, 0.5, 0.6, 0.7]),
]

METADATA_DOCS = [
    VectorStoreDocument(
        id="1",
        vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        data={"os": "windows", "category": "bug", "priority": 1},
    ),
    VectorStoreDocument(
        id="2",
        vector=[0.2, 0.3, 0.4, 0.5, 0.6],
        data={"os": "linux", "category": "feature", "priority": 2},
    ),
    VectorStoreDocument(
        id="3",
        vector=[0.3, 0.4, 0.5, 0.6, 0.7],
        data={"os": "windows", "category": "feature", "priority": 3},
    ),
]


# ---------------------------------------------------------------------------
# Module-scoped fixture: create table+index ONCE, drop after all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_store():
    """Create one SpannerVectorStore (table + vector index) for the whole module.

    Spanner vector index DDL takes 5–15 minutes; building it once per module
    avoids per-test DDL overhead.
    """
    cfg = _spanner_config()
    if not all(cfg.values()):
        pytest.skip("Spanner config not set")

    index_name = f"IntVec_{uuid4().hex[:8]}"
    database = _db(cfg)

    s = SpannerVectorStore(index_name=index_name, vector_size=VECTOR_SIZE, **cfg)
    s.connect()
    s.create_index()  # ~5–15 min — done once for the entire module

    yield s, index_name, database

    s.close()
    try:
        database.update_ddl([f"DROP TABLE `{index_name}`"]).result(timeout=300)
    except Exception as e:
        print(f"Warning: failed to drop `{index_name}`: {e}")


@pytest.fixture
def store(shared_store):
    """Return the shared store with all rows truncated (clean state per test)."""
    s, index_name, _ = shared_store
    s._database.execute_partitioned_dml(f"DELETE FROM `{index_name}` WHERE true")
    return s


@pytest.fixture
def store_with_docs(store):
    """Store pre-populated with METADATA_DOCS."""
    store.load_documents(METADATA_DOCS)
    return store


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

def test_load_and_count(store):
    assert store.count() == 0
    store.load_documents(SAMPLE_DOCS[:2])
    assert store.count() == 2


def test_load_documents_overwrites(store):
    """load_documents() replaces the entire table (overwrite semantics)."""
    store.load_documents(SAMPLE_DOCS)
    assert store.count() == 3
    store.load_documents(SAMPLE_DOCS[:1])
    assert store.count() == 1


def test_insert(store):
    """insert() adds a single document."""
    store.insert(VectorStoreDocument(id="x", vector=[0.1, 0.2, 0.3, 0.4, 0.5]))
    assert store.count() == 1


def test_search_by_id_found(store):
    store.load_documents(SAMPLE_DOCS)
    doc = store.search_by_id("1")
    assert doc.id == "1"
    assert doc.vector is not None
    assert len(doc.vector) == VECTOR_SIZE


def test_search_by_id_not_found_returns_stub(store):
    """SpannerVectorStore returns a stub doc (not raises) for missing IDs."""
    stub = store.search_by_id("nonexistent")
    assert stub.id == "nonexistent"
    assert stub.vector is None


def test_similarity_search_by_vector(store):
    store.load_documents(SAMPLE_DOCS)
    results = store.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4, 0.5], k=2)
    assert 1 <= len(results) <= 2
    assert results[0].document.id == "1"   # identical vector → score ≈ 1.0
    assert results[0].score > 0.99
    assert isinstance(results[0].score, float)


def test_similarity_search_k_limit(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=1
    )
    assert len(results) == 1


def test_similarity_search_by_text(store):
    store.load_documents(SAMPLE_DOCS)

    def mock_embedder(text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    results = store.similarity_search_by_text("test query", mock_embedder, k=2)
    assert 1 <= len(results) <= 2
    assert isinstance(results[0].score, float)


# ---------------------------------------------------------------------------
# remove / update
# ---------------------------------------------------------------------------

def test_remove(store_with_docs):
    assert store_with_docs.count() == 3
    store_with_docs.remove(["1", "2"])
    assert store_with_docs.count() == 1

    stub = store_with_docs.search_by_id("1")
    assert stub.vector is None

    remaining = store_with_docs.search_by_id("3")
    assert remaining.id == "3"


def test_remove_empty_list_is_noop(store_with_docs):
    store_with_docs.remove([])
    assert store_with_docs.count() == 3


def test_update(store_with_docs):
    store_with_docs.update(
        VectorStoreDocument(
            id="1",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            data={"os": "macos", "category": "bug", "priority": 1},
        )
    )
    doc = store_with_docs.search_by_id("1")
    assert doc.data.get("os") == "macos"


def test_update_sets_update_date(store_with_docs):
    doc_before = store_with_docs.search_by_id("1")
    assert doc_before.update_date is None

    store_with_docs.update(
        VectorStoreDocument(
            id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5], data={"os": "macos"}
        )
    )
    doc_after = store_with_docs.search_by_id("1")
    assert doc_after.update_date is not None


# ---------------------------------------------------------------------------
# select / include_vectors
# ---------------------------------------------------------------------------

def test_select_limits_fields(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=1, select=["os"]
    )
    data = results[0].document.data
    assert "os" in data
    assert "category" not in data
    assert "priority" not in data


def test_select_on_search_by_id(store_with_docs):
    doc = store_with_docs.search_by_id("1", select=["os"])
    assert "os" in doc.data
    assert "category" not in doc.data


def test_include_vectors_false(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=1, include_vectors=False
    )
    assert results[0].document.vector is None

    doc = store_with_docs.search_by_id("1", include_vectors=False)
    assert doc.vector is None


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------

def test_metadata_round_trips(store_with_docs):
    doc = store_with_docs.search_by_id("1")
    assert doc.data.get("os") == "windows"
    assert doc.data.get("category") == "bug"
    assert doc.data.get("priority") == 1


def test_fields_returned_in_search(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=1
    )
    assert results[0].document.data.get("os") == "windows"
    assert results[0].document.data.get("category") == "bug"
    assert results[0].document.data.get("priority") == 1


# ---------------------------------------------------------------------------
# FilterExpr — applied in-memory post-query
# ---------------------------------------------------------------------------

def test_filter_eq(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.os == "linux"
    )
    assert len(results) == 1
    assert results[0].document.id == "2"


def test_filter_ne(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.os != "linux"
    )
    assert len(results) == 2
    assert {r.document.id for r in results} == {"1", "3"}


def test_filter_gt_gte_lt_lte(store_with_docs):
    vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(store_with_docs.similarity_search_by_vector(vec, k=10, filters=F.priority > 1)) == 2
    assert len(store_with_docs.similarity_search_by_vector(vec, k=10, filters=F.priority >= 2)) == 2
    assert len(store_with_docs.similarity_search_by_vector(vec, k=10, filters=F.priority < 3)) == 2
    assert len(store_with_docs.similarity_search_by_vector(vec, k=10, filters=F.priority <= 1)) == 1


def test_filter_and(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        k=10,
        filters=(F.os == "windows") & (F.category == "feature"),
    )
    assert len(results) == 1
    assert results[0].document.id == "3"


def test_filter_or(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        k=10,
        filters=(F.os == "linux") | (F.category == "bug"),
    )
    assert len(results) == 2
    assert {r.document.id for r in results} == {"1", "2"}


def test_filter_not(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=~(F.os == "windows")
    )
    assert len(results) == 1
    assert results[0].document.id == "2"


def test_filter_in(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        k=10,
        filters=F.os.in_(["windows", "macos"]),
    )
    assert len(results) == 2
    assert {r.document.id for r in results} == {"1", "3"}


def test_filter_combined_preserves_score_order(store_with_docs):
    results = store_with_docs.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.category == "feature"
    )
    assert len(results) == 2
    assert results[0].score >= results[1].score


# ---------------------------------------------------------------------------
# Timestamps — create_date auto-set, components stored in data JSON
# ---------------------------------------------------------------------------

def test_create_date_auto_set(store):
    store.insert(VectorStoreDocument(id="dated", vector=[0.1, 0.2, 0.3, 0.4, 0.5]))
    doc = store.search_by_id("dated")
    assert doc.create_date is not None


def test_create_date_components_in_data(store):
    store.insert(
        VectorStoreDocument(
            id="comp",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            create_date="2024-03-15T14:30:00+00:00",
        )
    )
    doc = store.search_by_id("comp")
    assert doc.data.get("create_date_year") == 2024
    assert doc.data.get("create_date_month") == 3
    assert doc.data.get("create_date_month_name") == "March"
    assert doc.data.get("create_date_day") == 15
    assert doc.data.get("create_date_hour") == 14
    assert doc.data.get("create_date_quarter") == 1


def test_filter_by_timestamp_component(store):
    # 用 load_documents 一次性插入两条，避免 insert() 每次都 truncate 全表
    store.load_documents([
        VectorStoreDocument(
            id="dec",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            create_date="2024-12-25T10:00:00+00:00",
        ),
        VectorStoreDocument(
            id="mar",
            vector=[0.2, 0.3, 0.4, 0.5, 0.6],
            create_date="2024-03-15T10:00:00+00:00",
        ),
    ])
    results = store.similarity_search_by_vector(
        [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.create_date_month == 12
    )
    assert len(results) == 1
    assert results[0].document.id == "dec"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

def test_vector_store_factory(shared_store):
    """create_vector_store() with type='spanner' 产出可用的 SpannerVectorStore。

    复用 shared_store 的已有表（不重建 vector index），通过 update() 验证写入能力，
    通过 search_by_id() 验证读取能力。
    """
    from graphrag_vectors.index_schema import IndexSchema
    from graphrag_vectors.vector_store_config import VectorStoreConfig
    from graphrag_vectors.vector_store_factory import create_vector_store

    existing_store, index_name, _ = shared_store
    cfg = _spanner_config()

    # 用已有的 index_name，避免 DDL 建新 vector index
    config = VectorStoreConfig(type="spanner", vector_size=VECTOR_SIZE, **cfg)
    schema = IndexSchema(index_name=index_name, vector_size=VECTOR_SIZE)
    vs = create_vector_store(config, schema)
    vs.connect()
    try:
        # 先清空，再写入两行
        vs._database.execute_partitioned_dml(f"DELETE FROM `{index_name}` WHERE true")
        vs.update(VectorStoreDocument(id="f1", vector=SAMPLE_DOCS[0].vector, data={"src": "factory"}))
        vs.update(VectorStoreDocument(id="f2", vector=SAMPLE_DOCS[1].vector, data={"src": "factory"}))
        assert vs.count() == 2
        doc = vs.search_by_id("f1")
        assert doc.id == "f1"
        assert doc.data.get("src") == "factory"
    finally:
        vs.close()
