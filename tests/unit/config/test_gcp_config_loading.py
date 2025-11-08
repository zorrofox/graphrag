# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import pytest
from graphrag.config.models.storage_config import StorageConfig
from graphrag.config.models.vector_store_config import VectorStoreConfig
from graphrag.config.enums import StorageType, VectorStoreType

def test_storage_config_loads_gcs_fields():
    """Test that StorageConfig can load GCS-specific fields."""
    config_data = {
        "type": StorageType.gcs,
        "bucket_name": "test-bucket",
        "base_dir": "test-dir"
    }
    # Pydantic by default ignores extra fields unless configured otherwise.
    # We need to ensure they are NOT ignored and actually stored.
    config = StorageConfig(**config_data)
    assert config.type == StorageType.gcs
    assert hasattr(config, "bucket_name"), "StorageConfig missing bucket_name field"
    assert config.bucket_name == "test-bucket"

def test_storage_config_loads_spanner_fields():
    """Test that StorageConfig can load Spanner-specific fields."""
    config_data = {
        "type": StorageType.spanner,
        "project_id": "test-project",
        "instance_id": "test-instance",
        "database_id": "test-database",
        "table_prefix": "test-prefix"
    }
    config = StorageConfig(**config_data)
    assert config.type == StorageType.spanner
    assert hasattr(config, "project_id"), "StorageConfig missing project_id field"
    assert config.project_id == "test-project"
    assert config.instance_id == "test-instance"
    assert config.database_id == "test-database"
    assert config.table_prefix == "test-prefix"

def test_storage_config_defaults():
    """Test that new GCP fields default to None if not provided."""
    # Minimum required fields for StorageConfig might depend on type, but let's try generic
    config = StorageConfig(type=StorageType.memory)
    assert config.bucket_name is None
    assert config.project_id is None
    assert config.instance_id is None
    assert config.database_id is None
    assert config.table_prefix is None

def test_storage_config_partial_spanner():
    """Test that we can load a partial Spanner config (validation happens at runtime)."""
    config_data = {
        "type": StorageType.spanner,
        "project_id": "my-project"
        # Missing instance_id, database_id
    }
    config = StorageConfig(**config_data)
    assert config.type == StorageType.spanner
    assert config.project_id == "my-project"
    assert config.instance_id is None
    assert config.database_id is None

def test_vector_store_config_loads_spanner_fields():
    """Test that VectorStoreConfig can load Spanner-specific fields."""
    config_data = {
        "type": VectorStoreType.Spanner,
        "project_id": "test-project",
        "instance_id": "test-instance",
        "database_id": "test-database"
    }
    config = VectorStoreConfig(**config_data)
    assert config.type == VectorStoreType.Spanner
    assert hasattr(config, "project_id"), "VectorStoreConfig missing project_id field"
    assert config.project_id == "test-project"
    assert config.instance_id == "test-instance"
    assert config.database_id == "test-database"

def test_vector_store_config_defaults():
    """Test that new GCP fields default to None in VectorStoreConfig."""
    # LanceDB usually requires db_uri, but let's see if we can instantiate without it for this test
    # or use a type that doesn't require extra validation immediately.
    # Actually VectorStoreConfig has validators that might run.
    config_data = {"type": VectorStoreType.AzureAISearch, "url": "http://fake", "api_key": "fake"}
    config = VectorStoreConfig(**config_data)
    assert config.project_id is None
    assert config.instance_id is None
    assert config.database_id is None
