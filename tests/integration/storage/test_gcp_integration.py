# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import os
import pytest
import pandas as pd
from uuid import uuid4
from google.cloud import spanner

from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage
from graphrag.storage.spanner_pipeline_storage import SpannerPipelineStorage
from graphrag.vector_stores.spanner import SpannerVectorStore
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.config.models.storage_config import StorageConfig
from graphrag.storage.factory import StorageFactory
from graphrag.config.enums import StorageType

# Only run if explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("GRAPHRAG_GCP_INTEGRATION_TEST"),
    reason="GCP integration tests not enabled",
)

@pytest.fixture
def gcs_bucket():
    return os.environ.get("GCS_BUCKET_NAME")

@pytest.fixture
def spanner_config():
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID"),
        "instance_id": os.environ.get("SPANNER_INSTANCE_ID"),
        "database_id": os.environ.get("SPANNER_DATABASE_ID"),
    }

@pytest.fixture(scope="module")
def setup_spanner_tables():
    project_id = os.environ.get("GCP_PROJECT_ID")
    instance_id = os.environ.get("SPANNER_INSTANCE_ID")
    database_id = os.environ.get("SPANNER_DATABASE_ID")

    if not all([project_id, instance_id, database_id]):
        return

    client = spanner.Client(project=project_id)
    instance = client.instance(instance_id)
    database = instance.database(database_id)

    ddl = [
        """CREATE TABLE IF NOT EXISTS IntegrationTestBlobs (
            key STRING(MAX) NOT NULL,
            value BYTES(MAX),
            updated_at TIMESTAMP OPTIONS (allow_commit_timestamp=true)
        ) PRIMARY KEY (key)""",
        """CREATE TABLE IF NOT EXISTS TestVectorTable (
            id STRING(MAX) NOT NULL,
            text STRING(MAX),
            vector ARRAY<FLOAT64>,
            attributes JSON
        ) PRIMARY KEY (id)""",
        """CREATE TABLE IF NOT EXISTS FactoryTest_Blobs (
            key STRING(MAX) NOT NULL,
            value BYTES(MAX),
            updated_at TIMESTAMP OPTIONS (allow_commit_timestamp=true)
        ) PRIMARY KEY (key)"""
    ]

    try:
        operation = database.update_ddl(ddl)
        operation.result(timeout=300) # Wait for DDL to complete
    except Exception as e:
        print(f"Warning: Failed to update DDL, tables might already exist or permissions missing: {e}")

    # Cleanup before tests using transaction for strong consistency
    try:
        def delete_all(tx):
            tx.execute_update("DELETE FROM IntegrationTestBlobs WHERE true")
            tx.execute_update("DELETE FROM TestVectorTable WHERE true")
            tx.execute_update("DELETE FROM FactoryTest_Blobs WHERE true")
        database.run_in_transaction(delete_all)
    except Exception as e:
        print(f"Warning: Initial cleanup failed: {e}")

@pytest.mark.asyncio
async def test_gcs_storage_integration(gcs_bucket):
    if not gcs_bucket:
        pytest.skip("GCS_BUCKET_NAME not set")
    
    storage = GCSPipelineStorage(bucket_name=gcs_bucket, base_dir=f"integration-test-{uuid4()}")
    
    key = "test-file.txt"
    content = "integration test content"
    await storage.set(key, content)
    
    assert await storage.has(key)
    assert await storage.get(key) == content
    
    await storage.delete(key)
    assert not await storage.has(key)

@pytest.mark.asyncio
async def test_spanner_storage_integration(spanner_config, setup_spanner_tables):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # Use fixed prefix that maps to IntegrationTestBlobs
    table_prefix = "IntegrationTest"
    storage = SpannerPipelineStorage(**spanner_config, table_prefix=table_prefix)
    
    # Test blob storage (uses IntegrationTestBlobs table)
    key = f"test-blob-{uuid4()}"
    value = b"blob-content"
    try:
        await storage.set(key, value)
        assert await storage.has(key)
        assert await storage.get(key, as_bytes=True) == value
        await storage.delete(key)
        assert not await storage.has(key)
    except Exception as e:
        pytest.fail(f"Spanner blob test failed: {e}")

@pytest.mark.asyncio
async def test_spanner_vector_store_integration(spanner_config, setup_spanner_tables):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")
        
    index_name = "TestVectorTable"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3
    )
    
    store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
    store.connect()
    
    doc_id = f"doc-{uuid4()}"
    docs = [
        VectorStoreDocument(id=doc_id, text="integration test", vector=[0.1, 0.2, 0.3], attributes={"test": True})
    ]
    
    try:
        store.load_documents(docs)
        
        # Search by ID
        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id
        assert retrieved.text == "integration test"
        assert retrieved.attributes == {"test": True}
        
        # Vector search
        results = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=1)
        assert len(results) > 0
        assert results[0].document.id == doc_id
        # Score should be close to 1.0 for identical vector
        assert results[0].score > 0.99

    except Exception as e:
        pytest.fail(f"Spanner vector store test failed: {e}")

@pytest.mark.asyncio
async def test_spanner_table_storage_integration(spanner_config):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # 1. Setup: Create a table with complex types (simulating create_final_text_units)
    # Use a unique name to avoid conflicts if multiple tests run
    table_name = f"IntegrationTest_TextUnits_{uuid4().hex[:8]}"
    ddl = f"""
        CREATE TABLE {table_name} (
            id STRING(MAX) NOT NULL,
            text STRING(MAX),
            n_tokens INT64,
            document_ids ARRAY<STRING(MAX)>,
            attributes JSON
        ) PRIMARY KEY (id)
    """
    
    client = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])
    
    try:
        op = database.update_ddl([ddl])
        op.result(timeout=300) # Wait for table creation
    except Exception as e:
        pytest.fail(f"Failed to create test table {table_name}: {e}")

    try:
        # 2. Test: Write DataFrame to the table
        # Use empty prefix so we can use the exact table name we just created
        storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

        df = pd.DataFrame([
            {
                "id": "unit1",
                "text": "Sample text content",
                "n_tokens": 3,
                "document_ids": ["doc1", "doc2"],
                "attributes": {"source": "file1.txt", "page": 1}
            },
            {
                "id": "unit2",
                "text": "Another sample",
                "n_tokens": 2,
                "document_ids": ["doc3"],
                "attributes": None # Test null handling for JSON
            },
            {
                "id": "unit3",
                "text": "Empty list for JSON column",
                "n_tokens": 0,
                "document_ids": [],
                "attributes": [] # THIS IS THE CRITICAL TEST CASE for Expected JSON error
            }
        ])

        await storage.set_table(table_name, df)

        # 3. Verify: Read it back
        assert await storage.has_table(table_name)
        loaded_df = await storage.load_table(table_name)
        
        assert len(loaded_df) == 3
        # Sort by id to ensure consistent comparison
        loaded_df = loaded_df.sort_values("id").reset_index(drop=True)
        
        assert loaded_df.iloc[0]["id"] == "unit1"
        assert loaded_df.iloc[0]["n_tokens"] == 3
        # Spanner client returns lists for ARRAY
        assert loaded_df.iloc[0]["document_ids"] == ["doc1", "doc2"]
        # Spanner client returns dicts for JSON
        assert loaded_df.iloc[0]["attributes"] == {"source": "file1.txt", "page": 1}
        
        assert loaded_df.iloc[1]["id"] == "unit2"
        assert loaded_df.iloc[1]["attributes"] is None

        assert loaded_df.iloc[2]["id"] == "unit3"
        assert loaded_df.iloc[2]["document_ids"] == []
        
        # Handle Spanner JsonObject wrapper
        actual_attributes = loaded_df.iloc[2]["attributes"]
        if type(actual_attributes).__name__ == 'JsonObject':
             assert list(actual_attributes) == []
        else:
             assert actual_attributes == []

    finally:
        # 4. Cleanup: Drop the table
        try:
            op = database.update_ddl([f"DROP TABLE {table_name}"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop table {table_name}: {e}")

@pytest.mark.asyncio
async def test_spanner_auto_table_creation(spanner_config):
    """Test that SpannerPipelineStorage automatically creates tables if they don't exist."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # 1. Setup: Define a unique table name that definitely doesn't exist
    table_name = f"AutoCreateTest_{uuid4().hex[:8]}"
    
    client = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    # Ensure it doesn't exist (just in case of extremely unlikely collision)
    try:
        op = database.update_ddl([f"DROP TABLE {table_name}"])
        op.result(timeout=60)
    except Exception:
        pass # Expected if it doesn't exist

    try:
        # 2. Test: Write DataFrame to the non-existent table
        storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

        df = pd.DataFrame([
            {"id": "row1", "name": "Alice", "score": 95.5, "active": True, "tags": ["a", "b"]},
            {"id": "row2", "name": "Bob", "score": 80.0, "active": False, "tags": []}
        ])

        # This should trigger auto-creation
        await storage.set_table(table_name, df)

        # 3. Verify: Read it back
        assert await storage.has_table(table_name)
        loaded_df = await storage.load_table(table_name)
        
        assert len(loaded_df) == 2
        loaded_df = loaded_df.sort_values("id").reset_index(drop=True)
        
        assert loaded_df.iloc[0]["id"] == "row1"
        assert loaded_df.iloc[0]["name"] == "Alice"
        assert loaded_df.iloc[0]["score"] == 95.5
        assert loaded_df.iloc[0]["active"] == True
        # Note: Our inference might map ["a", "b"] to JSON or ARRAY<STRING>.
        # Spanner client returns list for both JSON array and ARRAY type.
        assert loaded_df.iloc[0]["tags"] == ["a", "b"]

    finally:
        # 4. Cleanup: Drop the table
        try:
            op = database.update_ddl([f"DROP TABLE {table_name}"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop auto-created table {table_name}: {e}")

@pytest.mark.asyncio
async def test_spanner_schema_evolution(spanner_config):
    """Test that SpannerPipelineStorage automatically adds new columns."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    table_name = f"SchemaEvoTest_{uuid4().hex[:8]}"
    client = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    # Ensure table doesn't exist
    try:
        op = database.update_ddl([f"DROP TABLE {table_name}"])
        op.result(timeout=60)
    except Exception:
        pass

    try:
        storage = SpannerPipelineStorage(**spanner_config, table_prefix="")

        # 1. Initial write (creates table)
        df1 = pd.DataFrame([{"id": "row1", "col1": "initial"}])
        await storage.set_table(table_name, df1)
        
        # Verify initial schema
        df_loaded = await storage.load_table(table_name)
        assert len(df_loaded.columns) == 2 # id, col1

        # 2. Second write with NEW column (triggers ALTER TABLE)
        df2 = pd.DataFrame([{"id": "row2", "col1": "updated", "new_col": 123}])
        await storage.set_table(table_name, df2)

        # 3. Verify evolved schema and data
        df_loaded = await storage.load_table(table_name)
        assert len(df_loaded.columns) == 3 # id, col1, new_col
        assert "new_col" in df_loaded.columns
        
        df_loaded = df_loaded.sort_values("id").reset_index(drop=True)
        # Row 1 should have null for new_col
        assert df_loaded.iloc[0]["id"] == "row1"
        assert pd.isna(df_loaded.iloc[0]["new_col"])
        # Row 2 should have value
        assert df_loaded.iloc[1]["id"] == "row2"
        assert df_loaded.iloc[1]["new_col"] == 123

    finally:
        try:
            op = database.update_ddl([f"DROP TABLE {table_name}"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop schema evolution test table {table_name}: {e}")

@pytest.mark.asyncio
async def test_spanner_storage_factory_integration(spanner_config, setup_spanner_tables):
    """Test that StorageFactory can create SpannerPipelineStorage from config."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # Create configuration object simulating loading from YAML/env
    config = StorageConfig(
        type=StorageType.spanner,
        project_id=spanner_config["project_id"],
        instance_id=spanner_config["instance_id"],
        database_id=spanner_config["database_id"],
        table_prefix="FactoryTest_"
    )

    # Use factory to create storage
    try:
        config_dict = config.model_dump()
    except AttributeError:
        config_dict = config.dict()

    storage = StorageFactory.create_storage(
        storage_type=config.type,
        kwargs=config_dict
    )
    
    assert isinstance(storage, SpannerPipelineStorage)
    
    # Verify it actually works
    key = f"factory-test-{uuid4()}"
    value = b"factory-content"
    try:
        await storage.set(key, value)
        assert await storage.get(key, as_bytes=True) == value
    finally:
        try:
            await storage.delete(key)
        except Exception:
            pass

@pytest.mark.asyncio
async def test_spanner_load_empty_table_columns(spanner_config):
    """Test loading an empty table returns correct columns."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    storage = SpannerPipelineStorage(**spanner_config, table_prefix="")
    table_name = f"EmptyTestTable_{uuid4().hex[:8]}"
    
    client = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    try:
        # Create table with some columns
        df = pd.DataFrame({"id": ["1"], "col1": ["a"], "col2": [1]})
        await storage.set_table(table_name, df)
        
        # Clear table
        database.execute_partitioned_dml(f"DELETE FROM {table_name} WHERE true")
        
        # Load empty table
        loaded_df = await storage.load_table(table_name)
        
        assert len(loaded_df) == 0
        assert set(loaded_df.columns) == {"id", "col1", "col2"}
        
    finally:
        try:
            op = database.update_ddl([f"DROP TABLE {table_name}"])
            op.result(timeout=300)
        except Exception:
            pass

@pytest.mark.asyncio
async def test_spanner_vector_store_auto_creation(spanner_config):
    """Test that SpannerVectorStore automatically creates the table if it doesn't exist."""
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # 1. Setup: Define a unique table name
    index_name = f"AutoVectorTest_{uuid4().hex[:8]}"
    config = VectorStoreSchemaConfig(
        index_name=index_name,
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3
    )
    
    client = spanner.Client(project=spanner_config["project_id"])
    instance = client.instance(spanner_config["instance_id"])
    database = instance.database(spanner_config["database_id"])

    # Ensure it doesn't exist
    try:
        op = database.update_ddl([f"DROP TABLE {index_name}"])
        op.result(timeout=60)
    except Exception:
        pass

    try:
        # 2. Test: Initialize store and load documents (triggers auto-creation)
        store = SpannerVectorStore(vector_store_schema_config=config, **spanner_config)
        store.connect()

        doc_id = f"doc-{uuid4()}"
        docs = [
            VectorStoreDocument(id=doc_id, text="auto creation test", vector=[0.1, 0.2, 0.3], attributes={"test": True})
        ]
        
        # This should trigger auto-creation and then insert
        store.load_documents(docs)

        # 3. Verify: Read it back
        retrieved = store.search_by_id(doc_id)
        assert retrieved.id == doc_id
        assert retrieved.text == "auto creation test"
        assert retrieved.vector == [0.1, 0.2, 0.3]
        # Spanner client returns dict for JSON
        assert retrieved.attributes == {"test": True}

        # 4. Verify Index Existence
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_NAME = @table_name AND INDEX_TYPE = 'VECTOR'",
                params={"table_name": index_name},
                param_types={"table_name": spanner.param_types.STRING}
            )
            indexes = [row[0] for row in results]
            assert f"{index_name}_VectorIndex" in indexes

    finally:
        # 5. Cleanup: Drop the table (this also drops the index)
        try:
            op = database.update_ddl([f"DROP TABLE {index_name}"])
            op.result(timeout=300)
        except Exception as e:
            print(f"Warning: Failed to drop auto-created vector table {index_name}: {e}")
