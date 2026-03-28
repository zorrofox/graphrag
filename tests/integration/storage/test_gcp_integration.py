# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for GCP storage backends (GCS and Spanner).

Run with real GCP credentials:

    export GRAPHRAG_GCP_INTEGRATION_TEST=1
    export GCS_BUCKET_NAME=your-bucket
    export GCP_PROJECT_ID=your-project
    export SPANNER_INSTANCE_ID=your-instance
    export SPANNER_DATABASE_ID=your-database
    uv run poe test_integration
"""

import os
import re
import pytest
import pandas as pd
from uuid import uuid4

from google.cloud import spanner

from graphrag.cache.factory import CacheFactory
from graphrag.cache.json_pipeline_cache import JsonPipelineCache
from graphrag.config.enums import CacheType, StorageType
from graphrag.config.models.storage_config import StorageConfig
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.storage.factory import StorageFactory
from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage
from graphrag.storage.spanner_pipeline_storage import SpannerPipelineStorage
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.vector_stores.spanner import SpannerVectorStore

# ---------------------------------------------------------------------------
# Skip gate — set GRAPHRAG_GCP_INTEGRATION_TEST=1 to enable
# ---------------------------------------------------------------------------

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


@pytest.fixture(scope="module")
def setup_spanner_tables():
    """Pre-create shared Spanner tables used by multiple tests and clean them."""
    project_id  = os.environ.get("GCP_PROJECT_ID")
    instance_id = os.environ.get("SPANNER_INSTANCE_ID")
    database_id = os.environ.get("SPANNER_DATABASE_ID")

    if not all([project_id, instance_id, database_id]):
        return

    client   = spanner.Client(project=project_id)
    instance = client.instance(instance_id)
    database = instance.database(database_id)

    # Create tables used by factory / blob integration tests.
    # TestVectorTable uses vector_length=>3 so it can carry a vector index.
    ddl = [
        """CREATE TABLE IF NOT EXISTS `IntegrationTestBlobs` (
            `key`        STRING(MAX) NOT NULL,
            `value`      BYTES(MAX),
            `updated_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true)
        ) PRIMARY KEY (`key`)""",
        """CREATE TABLE IF NOT EXISTS `TestVectorTable` (
            `id`         STRING(MAX) NOT NULL,
            `text`       STRING(MAX),
            `vector`     ARRAY<FLOAT64>(vector_length=>3),
            `attributes` JSON
        ) PRIMARY KEY (`id`)""",
        """CREATE VECTOR INDEX IF NOT EXISTS `TestVectorTable_VectorIndex`
            ON `TestVectorTable`(`vector`)
            WHERE `vector` IS NOT NULL
            OPTIONS (distance_type = 'COSINE')""",
        """CREATE TABLE IF NOT EXISTS `FactoryTest_Blobs` (
            `key`        STRING(MAX) NOT NULL,
            `value`      BYTES(MAX),
            `updated_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true)
        ) PRIMARY KEY (`key`)""",
    ]

    try:
        operation = database.update_ddl(ddl)
        operation.result(timeout=600)
    except Exception as e:
        print(f"Warning: DDL setup failed (tables may already exist): {e}")

    # Clean pre-existing data so tests start from a known state.
    try:
        def _delete_all(tx):
            tx.execute_update("DELETE FROM `IntegrationTestBlobs` WHERE true")
            tx.execute_update("DELETE FROM `TestVectorTable` WHERE true")
            tx.execute_update("DELETE FROM `FactoryTest_Blobs` WHERE true")

        database.run_in_transaction(_delete_all)
    except Exception as e:
        print(f"Warning: Initial cleanup failed: {e}")


# ---------------------------------------------------------------------------
# GCS — blob CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gcs_storage_integration(gcs_bucket):
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    storage = GCSPipelineStorage(
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
    """child() must reuse the parent client; find() must match by pattern and file_filter."""
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    base_dir = f"it-child-{uuid4().hex[:8]}"
    parent = GCSPipelineStorage(bucket_name=gcs_bucket, base_dir=base_dir)
    child  = parent.child("reports")

    try:
        # Client sharing
        assert child._client is parent._client, "child() must reuse the parent GCS client"
        assert child._base_dir == f"{base_dir}/reports"

        # Write files into the child directory
        await child.set("2024-01-report.txt", "report jan")
        await child.set("2024-06-report.txt", "report jun")
        await child.set("data.csv", "csv data")

        # find() by extension
        txt_results = list(child.find(re.compile(r".*\.txt$")))
        assert len(txt_results) == 2
        txt_names = {r[0] for r in txt_results}
        assert "2024-01-report.txt" in txt_names
        assert "2024-06-report.txt" in txt_names

        # find() with file_filter on named capture group
        pattern_with_group = re.compile(r"(?P<month>\d{4}-\d{2})-report\.txt")
        jan_results = list(child.find(pattern_with_group, file_filter={"month": "2024-01"}))
        assert len(jan_results) == 1
        assert jan_results[0][0] == "2024-01-report.txt"
        assert jan_results[0][1]["month"] == "2024-01"

        # keys() returns all three files
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

    storage = GCSPipelineStorage(
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
# GCS — cache factory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gcs_cache_factory_integration(gcs_bucket):
    """CacheFactory must produce a working GCS-backed JsonPipelineCache."""
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")

    base_dir = f"it-cache-{uuid4().hex[:8]}"
    cache = CacheFactory.create_cache(
        CacheType.gcs,
        {"bucket_name": gcs_bucket, "base_dir": base_dir},
    )
    assert isinstance(cache, JsonPipelineCache)

    key   = "test-cache-item"
    value = {"data": "test content"}
    try:
        await cache.set(key, value)
        assert await cache.has(key)
        assert await cache.get(key) == value
    finally:
        try:
            await cache._storage.delete(key)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Spanner — blob CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_storage_integration(spanner_config, setup_spanner_tables):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    storage = SpannerPipelineStorage(
        **spanner_config, table_prefix="IntegrationTest"
    )
    key   = f"test-blob-{uuid4().hex[:8]}"
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
    """child() must produce a sibling storage with a stacked table prefix."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    unique = uuid4().hex[:8]
    parent = SpannerPipelineStorage(
        **spanner_config, table_prefix=f"Parent_{unique}_"
    )
    child = parent.child("reports")

    try:
        # Prefix stacking
        assert child._table_prefix == f"Parent_{unique}_reports_"
        assert child._blob_table   == f"Parent_{unique}_reports_Blobs"

        # Connection params are inherited
        assert child._project_id  == spanner_config["project_id"]
        assert child._instance_id == spanner_config["instance_id"]
        assert child._database_id == spanner_config["database_id"]

        # Blobs written via child are invisible to parent
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
    """find() with pattern and file_filter, and keys() must work over Spanner blobs."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    unique  = uuid4().hex[:8]
    storage = SpannerPipelineStorage(
        **spanner_config, table_prefix=f"FindTest_{unique}_"
    )
    blob_keys = ["file1.parquet", "file2.parquet", "report.json", "summary.txt"]

    try:
        for k in blob_keys:
            await storage.set(k, b"content")

        # keys() returns all blobs
        all_keys = storage.keys()
        assert set(all_keys) == set(blob_keys)

        # find() by extension
        parquet = list(storage.find(re.compile(r".*\.parquet$")))
        assert len(parquet) == 2
        names = {r[0] for r in parquet}
        assert "file1.parquet" in names
        assert "file2.parquet" in names

        # find() with named-group file_filter
        pattern = re.compile(r"(?P<stem>file\d+)\.parquet")
        filtered = list(storage.find(pattern, file_filter={"stem": "file1"}))
        assert len(filtered) == 1
        assert filtered[0][0] == "file1.parquet"
        assert filtered[0][1]["stem"] == "file1"

        # find() with max_count
        limited = list(storage.find(re.compile(r".*\.parquet$"), max_count=1))
        assert len(limited) == 1

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
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    # Pre-create the table with a known schema
    ddl = f"""
        CREATE TABLE `{table_name}` (
            `id`           STRING(MAX) NOT NULL,
            `text`         STRING(MAX),
            `n_tokens`     INT64,
            `document_ids` ARRAY<STRING(MAX)>,
            `attributes`   JSON
        ) PRIMARY KEY (`id`)
    """
    try:
        op = database.update_ddl([ddl])
        op.result(timeout=300)
    except Exception as e:
        pytest.fail(f"Failed to create test table `{table_name}`: {e}")

    try:
        storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

        df = pd.DataFrame([
            {
                "id":           "unit1",
                "text":         "Sample text",
                "n_tokens":     3,
                "document_ids": ["doc1", "doc2"],
                "attributes":   {"source": "file1.txt", "page": 1},
            },
            {
                "id":           "unit2",
                "text":         "Another sample",
                "n_tokens":     2,
                "document_ids": ["doc3"],
                "attributes":   None,
            },
            {
                "id":           "unit3",
                "text":         "Empty list edge case",
                "n_tokens":     0,
                "document_ids": [],
                "attributes":   [],
            },
        ])

        await storage.set_table(table_name, df)

        assert await storage.has_table(table_name)
        loaded = (await storage.load_table(table_name)).sort_values("id").reset_index(drop=True)

        assert len(loaded) == 3
        assert loaded.iloc[0]["id"] == "unit1"
        assert loaded.iloc[0]["n_tokens"] == 3
        assert loaded.iloc[0]["document_ids"] == ["doc1", "doc2"]
        assert loaded.iloc[0]["attributes"] == {"source": "file1.txt", "page": 1}
        assert loaded.iloc[1]["attributes"] is None
        assert loaded.iloc[2]["document_ids"] == []

    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{table_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_auto_table_creation(spanner_config):
    """set_table() must auto-create the Spanner table when it does not exist."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"AutoCreate_{uuid4().hex[:8]}"
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    storage = SpannerPipelineStorage(**spanner_config, table_prefix="")
    df = pd.DataFrame([
        {"id": "row1", "name": "Alice", "score": 95.5, "active": True,  "tags": ["a", "b"]},
        {"id": "row2", "name": "Bob",   "score": 80.0, "active": False, "tags": []},
    ])

    try:
        await storage.set_table(table_name, df)
        assert await storage.has_table(table_name)

        loaded = (await storage.load_table(table_name)).sort_values("id").reset_index(drop=True)
        assert len(loaded) == 2
        assert loaded.iloc[0]["name"] == "Alice"
        assert loaded.iloc[0]["score"] == 95.5
        assert loaded.iloc[0]["tags"] == ["a", "b"]
    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{table_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_schema_evolution(spanner_config):
    """set_table() must automatically ALTER the table when new columns appear."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"SchemaEvo_{uuid4().hex[:8]}"
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

    try:
        # Initial write — creates table with 2 columns
        await storage.set_table(table_name, pd.DataFrame([{"id": "r1", "col1": "initial"}]))
        loaded = await storage.load_table(table_name)
        assert set(loaded.columns) == {"id", "col1"}

        # Second write adds a new column — triggers ALTER TABLE
        await storage.set_table(table_name, pd.DataFrame([{"id": "r2", "col1": "v2", "new_col": 123}]))
        loaded = await storage.load_table(table_name)
        assert "new_col" in loaded.columns

        loaded = loaded.sort_values("id").reset_index(drop=True)
        assert pd.isna(loaded.iloc[0]["new_col"])   # r1 gets NULL for new_col
        assert loaded.iloc[1]["new_col"] == 123
    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{table_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


@pytest.mark.asyncio
async def test_spanner_load_empty_table_columns(spanner_config):
    """load_table() on an empty table must return a DataFrame with the correct columns."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"EmptyTest_{uuid4().hex[:8]}"
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

    try:
        await storage.set_table(table_name, pd.DataFrame({"id": ["1"], "col1": ["a"], "col2": [1]}))
        # Clear all rows via partitioned DML
        database.execute_partitioned_dml(f"DELETE FROM `{table_name}` WHERE true")

        loaded = await storage.load_table(table_name)
        assert len(loaded) == 0
        assert set(loaded.columns) == {"id", "col1", "col2"}
    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{table_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{table_name}`: {e}")
        storage.close()


# ---------------------------------------------------------------------------
# Spanner — vector store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_vector_store_integration(spanner_config, setup_spanner_tables):
    """End-to-end: load document, search by id, vector similarity search."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    config = VectorStoreSchemaConfig(
        index_name="TestVectorTable",
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3,
    )
    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()

    doc_id = f"doc-{uuid4().hex[:8]}"
    docs = [
        VectorStoreDocument(
            id=doc_id,
            text="integration test",
            vector=[0.1, 0.2, 0.3],
            attributes={"test": True},
        )
    ]
    try:
        store.load_documents(docs)

        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id
        assert retrieved.text == "integration test"
        assert retrieved.attributes == {"test": True}

        results = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=1)
        assert len(results) > 0
        assert results[0].document.id == doc_id
        assert results[0].score > 0.99  # identical vector → score ≈ 1.0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_store_auto_creation(spanner_config):
    """load_documents() must auto-create the vector table + index when absent."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    index_name = f"AutoVec_{uuid4().hex[:8]}"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3,
    )
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()

    doc_id = f"doc-{uuid4().hex[:8]}"
    try:
        store.load_documents([
            VectorStoreDocument(
                id=doc_id, text="auto creation test",
                vector=[0.1, 0.2, 0.3], attributes={"test": True},
            )
        ])

        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id
        assert retrieved.attributes == {"test": True}

        # Verify vector index was created
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
            op = database.update_ddl([f"DROP TABLE `{index_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_overwrite_removes_stale_documents(spanner_config):
    """load_documents(overwrite=True) must remove docs not present in the new batch."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    index_name = f"OverwriteTest_{uuid4().hex[:8]}"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3,
    )
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()

    try:
        # First load — two documents
        store.load_documents([
            VectorStoreDocument(id="stale",   text="will be removed", vector=[0.9, 0.0, 0.0]),
            VectorStoreDocument(id="current", text="will stay",        vector=[0.1, 0.2, 0.3]),
        ], overwrite=True)

        # Second load — only one document, overwrite=True
        store.load_documents([
            VectorStoreDocument(id="current", text="updated", vector=[0.1, 0.2, 0.3]),
        ], overwrite=True)

        # "stale" must be gone
        gone = store.search_by_id("stale")
        assert gone.text is None, "Stale document must have been deleted by overwrite=True"

        # "current" must still exist with updated text
        kept = store.search_by_id("current")
        assert kept.id == "current"
        assert kept.text == "updated"

    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{index_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_overwrite_false_keeps_existing(spanner_config):
    """load_documents(overwrite=False) must not remove pre-existing documents."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    index_name = f"UpsertTest_{uuid4().hex[:8]}"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3,
    )
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()

    try:
        # Initial load
        store.load_documents([
            VectorStoreDocument(id="existing", text="original", vector=[0.1, 0.2, 0.3]),
        ], overwrite=True)

        # Second load with overwrite=False — must not delete "existing"
        store.load_documents([
            VectorStoreDocument(id="new", text="added", vector=[0.4, 0.5, 0.6]),
        ], overwrite=False)

        assert store.search_by_id("existing").text == "original"
        assert store.search_by_id("new").text == "added"

    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{index_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


@pytest.mark.asyncio
async def test_spanner_vector_auto_create_with_length(spanner_config):
    """The auto-created vector column must have the correct vector_length constraint."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    vector_size = 4
    index_name  = f"VecLen_{uuid4().hex[:8]}"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=vector_size,
    )
    client   = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()

    try:
        store.load_documents([
            VectorStoreDocument(
                id="v1", text="len test",
                vector=[0.1] * vector_size,
            )
        ])

        # Verify INFORMATION_SCHEMA records the correct vector_length
        with database.snapshot() as snap:
            rows = list(snap.execute_sql(
                "SELECT SPANNER_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_NAME = @t AND COLUMN_NAME = 'vector'",
                params={"t": index_name},
                param_types={"t": spanner.param_types.STRING},
            ))
        assert rows, "Column 'vector' not found in INFORMATION_SCHEMA"
        col_type = rows[0][0]
        assert f"vector_length=>{vector_size}" in col_type, (
            f"Expected 'vector_length=>{vector_size}' in column type, got: {col_type!r}"
        )

        # Document must be retrievable and searchable
        results = store.similarity_search_by_vector([0.1] * vector_size, k=1)
        assert len(results) == 1
        assert results[0].document.id == "v1"
    finally:
        try:
            op = database.update_ddl([f"DROP TABLE `{index_name}`"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop `{index_name}`: {e}")
        store.close()


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanner_storage_factory_integration(spanner_config, setup_spanner_tables):
    """StorageFactory must produce a working SpannerPipelineStorage from config."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    config = StorageConfig(
        type=StorageType.spanner,
        project_id=spanner_config["project_id"],
        instance_id=spanner_config["instance_id"],
        database_id=spanner_config["database_id"],
        table_prefix="FactoryTest_",
    )
    storage = StorageFactory.create_storage(
        storage_type=config.type,
        kwargs=config.model_dump(),
    )
    assert isinstance(storage, SpannerPipelineStorage)

    key   = f"factory-{uuid4().hex[:8]}"
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
