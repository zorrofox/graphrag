# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for GCP storage backends (GCS and Spanner) — v3 monorepo.

Run with real GCP credentials:

    export GRAPHRAG_GCP_INTEGRATION_TEST=1
    export GCS_BUCKET_NAME=your-bucket
    export GCP_PROJECT_ID=your-project
    export SPANNER_INSTANCE_ID=your-instance
    export SPANNER_DATABASE_ID=your-database
    uv run python -m pytest tests/integration/storage/test_gcp_integration.py -v
"""

import os
import re
from uuid import uuid4

import pandas as pd
import pytest
from google.cloud import spanner

from graphrag_cache.gcs_litellm_cache import GCSLiteLLMCache
from graphrag_storage.gcs_storage import GCSStorage
from graphrag_storage.spanner_storage import SpannerStorage
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage
from graphrag_storage.storage_type import StorageType
from graphrag_vectors.spanner import SpannerVectorStore
from graphrag_vectors.vector_store import VectorStoreDocument

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRAPHRAG_GCP_INTEGRATION_TEST"),
    reason="GCP integration tests not enabled (set GRAPHRAG_GCP_INTEGRATION_TEST=1)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gcs_bucket() -> str | None:
    return os.environ.get("GCS_BUCKET_NAME")


@pytest.fixture
def spanner_config() -> dict:
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID"),
        "instance_id": os.environ.get("SPANNER_INSTANCE_ID"),
        "database_id": os.environ.get("SPANNER_DATABASE_ID"),
    }


# ---------------------------------------------------------------------------
# GCS — blob CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gcs_storage_integration(gcs_bucket):
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    storage = GCSStorage(
        bucket_name=gcs_bucket,
        base_dir=f"integration-test-{uuid4().hex[:8]}",
    )
    key = "test-file.txt"
    content = "integration test content"

    try:
        await storage.set(key, content)
        assert await storage.has(key)
        assert await storage.get(key) == content
        await storage.delete(key)
        assert not await storage.has(key)
    finally:
        storage.close()


# ---------------------------------------------------------------------------
# GCS — child() client sharing, find(), keys(), clear()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gcs_child_and_find(gcs_bucket):
    """child() reuses the parent GCS client; find() matches by pattern."""
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    base_dir = f"it-child-{uuid4().hex[:8]}"
    parent = GCSStorage(bucket_name=gcs_bucket, base_dir=base_dir)
    child = parent.child("reports")

    try:
        assert child._client is parent._client, "child() must reuse the parent GCS client"
        assert child._base_dir == f"{base_dir}/reports"

        await child.set("2024-01-report.txt", "report jan")
        await child.set("2024-06-report.txt", "report jun")
        await child.set("data.csv", "csv data")

        # find() returns Iterator[str] in v3
        txt_results = list(child.find(re.compile(r".*\.txt$")))
        assert len(txt_results) == 2
        assert "2024-01-report.txt" in txt_results
        assert "2024-06-report.txt" in txt_results

        all_keys = child.keys()
        assert len(all_keys) == 3
        assert "data.csv" in all_keys

    finally:
        try:
            await child.clear()
        except Exception:
            pass
        parent.close()


@pytest.mark.asyncio
async def test_gcs_keys_and_clear(gcs_bucket):
    """keys() returns all objects; clear() deletes them all."""
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    storage = GCSStorage(
        bucket_name=gcs_bucket,
        base_dir=f"it-clear-{uuid4().hex[:8]}",
    )
    try:
        await storage.set("a.txt", "a")
        await storage.set("b.txt", "b")
        await storage.set("c.txt", "c")

        keys = storage.keys()
        assert len(keys) == 3
        assert set(keys) == {"a.txt", "b.txt", "c.txt"}

        await storage.clear()
        assert storage.keys() == []
    finally:
        try:
            await storage.clear()
        except Exception:
            pass
        storage.close()


# ---------------------------------------------------------------------------
# GCS — LiteLLM cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gcs_litellm_cache_integration(gcs_bucket):
    """GCSLiteLLMCache must persist and retrieve LLM response dicts via GCS."""
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    base_dir = f"it-litellm-{uuid4().hex[:8]}"
    cache = GCSLiteLLMCache(bucket_name=gcs_bucket, base_dir=base_dir)

    cache_key = f"test-llm-{uuid4().hex[:8]}"
    llm_response = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "model": "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    try:
        assert await cache.async_get_cache(cache_key) is None
        await cache.async_set_cache(cache_key, llm_response)
        retrieved = await cache.async_get_cache(cache_key)
        assert retrieved == llm_response
        assert await cache.async_get_cache("other-key") is None
    finally:
        try:
            await cache._storage.clear()
        except Exception:
            pass
        cache.close()


# ---------------------------------------------------------------------------
# Spanner — blob CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_storage_integration(spanner_config):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    storage = SpannerStorage(**spanner_config, table_prefix="IntegrationTest")
    key = f"test-blob-{uuid4().hex[:8]}"
    value = b"blob-content"
    try:
        await storage.set(key, value)
        assert await storage.has(key)
        assert await storage.get(key, as_bytes=True) == value
        await storage.delete(key)
        assert not await storage.has(key)
    finally:
        storage.close()


# ---------------------------------------------------------------------------
# Spanner — child() prefix isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_child_prefix_isolation(spanner_config):
    """child() produces a sibling storage with a stacked table prefix."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    unique = uuid4().hex[:8]
    parent = SpannerStorage(**spanner_config, table_prefix=f"Parent_{unique}_")
    child = parent.child("reports")

    try:
        assert child._table_prefix == f"Parent_{unique}_reports_"
        assert child._blob_table == f"Parent_{unique}_reports_Blobs"
        assert child._project_id == spanner_config["project_id"]
        assert child._instance_id == spanner_config["instance_id"]
        assert child._database_id == spanner_config["database_id"]

        key = f"child-blob-{uuid4().hex[:8]}"
        await child.set(key, b"child content")
        assert await child.has(key)
        assert await child.get(key, as_bytes=True) == b"child content"
        assert not await parent.has(key), "Parent blob table must be separate from child"
    finally:
        try:
            await child.clear()
        except Exception:
            pass
        child.close()
        parent.close()


# ---------------------------------------------------------------------------
# Spanner — find() and keys()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_find_and_keys(spanner_config):
    """find() with pattern and keys() over Spanner blobs."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    unique = uuid4().hex[:8]
    storage = SpannerStorage(**spanner_config, table_prefix=f"FindTest_{unique}_")
    blob_keys = ["file1.parquet", "file2.parquet", "report.json", "summary.txt"]

    try:
        for k in blob_keys:
            await storage.set(k, b"content")

        all_keys = storage.keys()
        assert set(all_keys) == set(blob_keys)

        # find() returns Iterator[str] in v3
        parquet = list(storage.find(re.compile(r".*\.parquet$")))
        assert len(parquet) == 2
        assert "file1.parquet" in parquet
        assert "file2.parquet" in parquet

    finally:
        for k in blob_keys:
            try:
                await storage.delete(k)
            except Exception:
                pass
        storage.close()


# ---------------------------------------------------------------------------
# Spanner — table storage (DataFrame write / read)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_table_storage_integration(spanner_config):
    """Write a DataFrame with complex types; verify round-trip fidelity."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"IntTest_TextUnits_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )

    ddl = (
        f"CREATE TABLE `{table_name}` ("
        " `id` STRING(MAX) NOT NULL,"
        " `text` STRING(MAX),"
        " `n_tokens` INT64,"
        " `document_ids` ARRAY<STRING(MAX)>,"
        " `attributes` JSON"
        f") PRIMARY KEY (`id`)"
    )
    try:
        database.update_ddl([ddl]).result(timeout=300)
    except Exception as e:
        pytest.fail(f"Failed to create test table `{table_name}`: {e}")

    try:
        storage = SpannerStorage(**spanner_config, table_prefix="")
        df = pd.DataFrame([
            {"id": "unit1", "text": "Sample text", "n_tokens": 3,
             "document_ids": ["doc1", "doc2"], "attributes": {"source": "file1.txt"}},
            {"id": "unit2", "text": "Another sample", "n_tokens": 2,
             "document_ids": ["doc3"], "attributes": None},
        ])

        await storage.set_table(table_name, df)

        assert await storage.has_table(table_name)
        loaded = (await storage.load_table(table_name)).sort_values("id").reset_index(drop=True)

        assert len(loaded) == 2
        assert loaded.iloc[0]["id"] == "unit1"
        assert loaded.iloc[0]["n_tokens"] == 3
        assert loaded.iloc[0]["document_ids"] == ["doc1", "doc2"]
        assert loaded.iloc[1]["attributes"] is None

    finally:
        try:
            database.update_ddl([f"DROP TABLE `{table_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_auto_table_creation(spanner_config):
    """set_table() auto-creates the Spanner table when it does not exist."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"AutoCreate_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )
    storage = SpannerStorage(**spanner_config, table_prefix="")
    df = pd.DataFrame([
        {"id": "row1", "name": "Alice", "score": 95.5, "active": True, "tags": ["a", "b"]},
        {"id": "row2", "name": "Bob", "score": 80.0, "active": False, "tags": []},
    ])

    try:
        await storage.set_table(table_name, df)
        assert await storage.has_table(table_name)
        loaded = (await storage.load_table(table_name)).sort_values("id").reset_index(drop=True)
        assert len(loaded) == 2
        assert loaded.iloc[0]["name"] == "Alice"
        assert loaded.iloc[0]["score"] == 95.5
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{table_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_schema_evolution(spanner_config):
    """set_table() auto-ALTERs the table when new columns appear."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"SchemaEvo_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )
    storage = SpannerStorage(**spanner_config, table_prefix="")

    try:
        await storage.set_table(table_name, pd.DataFrame([{"id": "r1", "col1": "initial"}]))
        loaded = await storage.load_table(table_name)
        assert set(loaded.columns) == {"id", "col1"}

        await storage.set_table(
            table_name, pd.DataFrame([{"id": "r2", "col1": "v2", "new_col": 123}])
        )
        loaded = await storage.load_table(table_name)
        assert "new_col" in loaded.columns
        loaded = loaded.sort_values("id").reset_index(drop=True)
        assert pd.isna(loaded.iloc[0]["new_col"])
        assert loaded.iloc[1]["new_col"] == 123
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{table_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_load_empty_table_columns(spanner_config):
    """load_table() on an empty table returns a DataFrame with correct columns."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"EmptyTest_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )
    storage = SpannerStorage(**spanner_config, table_prefix="")

    try:
        await storage.set_table(
            table_name, pd.DataFrame({"id": ["1"], "col1": ["a"], "col2": [1]})
        )
        database.execute_partitioned_dml(f"DELETE FROM `{table_name}` WHERE true")
        loaded = await storage.load_table(table_name)
        assert len(loaded) == 0
        assert set(loaded.columns) == {"id", "col1", "col2"}
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{table_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_load_table_pagination(spanner_config):
    """load_table(limit=N, offset=M) returns the correct slice of rows."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"PaginationTest_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )
    storage = SpannerStorage(**spanner_config, table_prefix="")
    df = pd.DataFrame({"id": [f"row{i:02d}" for i in range(15)], "val": list(range(15))})

    try:
        await storage.set_table(table_name, df)

        page1 = await storage.load_table(table_name, limit=5, offset=0)
        page2 = await storage.load_table(table_name, limit=5, offset=5)
        page3 = await storage.load_table(table_name, limit=5, offset=10)

        assert len(page1) == 5
        assert len(page2) == 5
        assert len(page3) == 5
        assert set(page1["id"]).isdisjoint(set(page2["id"]))

        full = await storage.load_table(table_name)
        assert len(full) == 15
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{table_name}`"]).result(timeout=120)
        except Exception as e:
            print(f"Warning: cleanup failed for `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_blob_creation_date_preserved_on_update(spanner_config):
    """created_at must be set on first write and not change on subsequent updates."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    unique = uuid4().hex[:8]
    storage = SpannerStorage(**spanner_config, table_prefix=f"CreatedAt_{unique}_")
    key = "test-blob.bin"

    try:
        await storage.set(key, b"original content")
        created_at_1 = await storage.get_creation_date(key)
        assert created_at_1, "created_at must be non-empty after first write"

        await storage.set(key, b"updated content")
        created_at_2 = await storage.get_creation_date(key)
        assert created_at_2 == created_at_1, (
            f"created_at must not change on update: {created_at_1!r} → {created_at_2!r}"
        )
        assert await storage.get(key, as_bytes=True) == b"updated content"
    finally:
        try:
            await storage.delete(key)
        except Exception:
            pass
        storage.close()


# ---------------------------------------------------------------------------
# Spanner — vector store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_vector_store_integration(spanner_config):
    """End-to-end: load document, search by id, vector similarity search."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    index_name = f"VecIntTest_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )

    store = SpannerVectorStore(
        index_name=index_name,
        vector_size=3,
        **spanner_config,
    )
    store.connect()

    doc_id = f"doc-{uuid4().hex[:8]}"
    docs = [
        VectorStoreDocument(
            id=doc_id,
            vector=[0.1, 0.2, 0.3],
            data={"text": "integration test", "category": "test"},
        )
    ]
    try:
        store.load_documents(docs)

        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id
        assert retrieved.data.get("text") == "integration test"

        results = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=1)
        assert len(results) > 0
        assert results[0].document.id == doc_id
        assert results[0].score > 0.99

        total = store.count()
        assert total >= 1
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{index_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_store_auto_creation(spanner_config):
    """load_documents() auto-creates the vector table + index when absent."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    index_name = f"AutoVec_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )

    store = SpannerVectorStore(index_name=index_name, vector_size=3, **spanner_config)
    store.connect()

    doc_id = f"doc-{uuid4().hex[:8]}"
    try:
        store.load_documents([
            VectorStoreDocument(id=doc_id, vector=[0.1, 0.2, 0.3], data={"label": "auto"})
        ])

        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id

        with database.snapshot() as snap:
            rows = list(snap.execute_sql(
                "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES "
                "WHERE TABLE_NAME = @t AND INDEX_TYPE = 'VECTOR'",
                params={"t": index_name},
                param_types={"t": spanner.param_types.STRING},
            ))
        index_names = [r[0] for r in rows]
        assert f"{index_name}_VectorIndex" in index_names
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{index_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_auto_create_with_length(spanner_config):
    """The auto-created vector column must have the correct vector_length constraint."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    vector_size = 4
    index_name = f"VecLen_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    database = client.instance(spanner_config["instance_id"]).database(
        spanner_config["database_id"]
    )

    store = SpannerVectorStore(index_name=index_name, vector_size=vector_size, **spanner_config)
    store.connect()

    try:
        store.load_documents([
            VectorStoreDocument(id="v1", vector=[0.1] * vector_size, data={})
        ])

        with database.snapshot() as snap:
            rows = list(snap.execute_sql(
                "SELECT SPANNER_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_NAME = @t AND COLUMN_NAME = 'vector'",
                params={"t": index_name},
                param_types={"t": spanner.param_types.STRING},
            ))
        assert rows
        col_type = rows[0][0]
        assert f"vector_length=>{vector_size}" in col_type, (
            f"Expected 'vector_length=>{vector_size}' in column type, got: {col_type!r}"
        )

        results = store.similarity_search_by_vector([0.1] * vector_size, k=1)
        assert len(results) == 1
        assert results[0].document.id == "v1"
    finally:
        try:
            database.update_ddl([f"DROP TABLE `{index_name}`"]).result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_storage_factory_integration(spanner_config):
    """create_storage() must produce a working SpannerStorage from config."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    config = StorageConfig(
        type=StorageType.Spanner,
        project_id=spanner_config["project_id"],
        instance_id=spanner_config["instance_id"],
        database_id=spanner_config["database_id"],
        table_prefix="FactoryTest_",
    )
    storage = create_storage(config)
    assert isinstance(storage, SpannerStorage)

    key = f"factory-{uuid4().hex[:8]}"
    value = b"factory-content"
    try:
        await storage.set(key, value)
        assert await storage.get(key, as_bytes=True) == value
    finally:
        try:
            await storage.delete(key)
        except Exception:
            pass
        storage.close()
