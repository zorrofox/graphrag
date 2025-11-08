# GraphRAG on Google Cloud Platform

This document outlines the plan and requirements for adapting GraphRAG to run natively on Google Cloud Platform services.

## Overview

The goal is to extend GraphRAG's storage and vector search capabilities to support Google Cloud services, specifically Google Cloud Storage (GCS) and Google Cloud Spanner.

## Key Objectives

1.  **GCS for Input/Output Storage**: Implement a `PipelineStorage` backend backed by Google Cloud Storage. This will allow the pipeline to read input files (text, CSV, etc.) and write output artifacts (Parquet files, reports) directly to GCS buckets.
2.  **Spanner for Tabular Output**: Implement a storage mechanism to write pipeline structured outputs (e.g., final documents, text units, community reports) directly to Google Cloud Spanner tables, rather than just as Parquet blobs.
3.  **Spanner for Vector Store**: Implement a `VectorStore` backend using Google Cloud Spanner's vector search capabilities (KNN).

## Technical Requirements

*   **Testing**:
    *   All new code must have comprehensive unit tests.
    *   Integration tests must be provided for GCS and Spanner implementations.
    *   Tests should use `unittest.mock` where appropriate to avoid requiring live GCP credentials for unit tests.
*   **Dependency Management**:
    *   Use a Python virtual environment (`venv`) for managing dependencies during development and testing.
    *   New dependencies (e.g., `google-cloud-storage`, `google-cloud-spanner`) should be added to `pyproject.toml`.

## Implementation Plan

### 1. Google Cloud Storage (GCS) Pipeline Storage

*   **Target**: `graphrag/storage/gcs_pipeline_storage.py`
*   **Inherits from**: `graphrag.storage.pipeline_storage.PipelineStorage`
*   **Description**: Implements standard blob storage operations (`get`, `set`, `find`, `delete`, etc.) using the GCS client library.

### 2. Google Cloud Spanner Pipeline Storage (Output)

*   **Target**: `graphrag/storage/spanner_pipeline_storage.py`
*   **Inherits from**: `graphrag.storage.pipeline_storage.PipelineStorage`
*   **Description**:
    *   Needs to handle generic blob storage (simulating a file system within Spanner, possibly inefficient but required for full `PipelineStorage` compliance if used generically).
    *   **Crucially**, it needs to handle structured table writes. Since the current `write_table_to_storage` utility forces Parquet conversion, we may need to:
        *   *Option A*: In `SpannerPipelineStorage.set`, detect `.parquet` keys, deserialize the Parquet bytes back to a DataFrame, and write rows to Spanner.
        *   *Option B (Preferred)*: Refactor `graphrag.utils.storage.write_table_to_storage` to check if the storage backend supports native DataFrame writes, bypassing Parquet conversion.

### 3. Google Cloud Spanner Vector Store

*   **Target**: `graphrag/vector_stores/spanner.py`
*   **Inherits from**: `graphrag.vector_stores.base.BaseVectorStore`
*   **Description**:
    *   Implements `load_documents` to write embeddings and metadata to a Spanner table.
    *   Implements `similarity_search_by_vector` using Spanner's `COSINE_DISTANCE` (or equivalent) in SQL queries.

## Development Workflow

1.  Set up `venv` and install dependencies.
2.  Implement GCS Storage + Unit Tests.
3.  Implement Spanner Vector Store + Unit Tests.
4.  Implement Spanner Output Storage + Unit Tests.
5.  Run Integration Tests.

## Progress Log

*   **[DATE]**: Created Python virtual environment (`.venv`).
*   **[DATE]**: Added `google-cloud-storage` and `google-cloud-spanner` to `pyproject.toml`.
*   **[DATE]**: Installed dependencies into `.venv`.
*   **[DATE]**: Implemented `GCSPipelineStorage` in `graphrag/storage/gcs_pipeline_storage.py`.
*   **[DATE]**: Implemented and passed unit tests for `GCSPipelineStorage` in `tests/unit/storage/test_gcs_pipeline_storage.py`.
*   **[DATE]**: Implemented `SpannerVectorStore` in `graphrag/vector_stores/spanner.py`.
*   **[DATE]**: Implemented and passed unit tests for `SpannerVectorStore` in `tests/unit/vector_stores/test_spanner.py`.
*   **[DATE]**: Refactored `graphrag/utils/storage.py` to support `set_table` and `load_table` hooks.
*   **[DATE]**: Implemented `SpannerPipelineStorage` in `graphrag/storage/spanner_pipeline_storage.py`.
*   **[DATE]**: Implemented and passed unit tests for `SpannerPipelineStorage` in `tests/unit/storage/test_spanner_pipeline_storage.py`.
*   **[DATE]**: Registered new storage and vector store types in factories and enums.
*   **[DATE]**: Created integration tests in `tests/integration/storage/test_gcp_integration.py` (skipped by default).
*   **[DATE]**: **Project Complete**.

## Architectural Decisions & Known Issues

### 1. Configuration Model Updates
To support GCP-specific configuration without relying on untyped `kwargs` passing throughout the system, we modified the core Pydantic models in `graphrag/config/models/`:
*   `StorageConfig`: Added `bucket_name`, `project_id`, `instance_id`, `database_id`, `table_prefix`.
*   `VectorStoreConfig`: Added `project_id`, `instance_id`, `database_id`.
This allows standard GraphRAG configuration loaders (YAML, environment variables) to correctly parse and validate these fields.

### 2. Spanner `BYTES` Encoding Workaround
The `google-cloud-spanner` Python client exhibits inconsistent behavior when writing to `BYTES` columns, sometimes failing to automatically Base64-encode `bytes` objects or incorrectly treating them as strings.
*   **Solution**: In `SpannerPipelineStorage.set`, we explicitly Base64-encode the data to an ASCII string and use a DML statement with the `FROM_BASE64()` SQL function to ensure correct server-side decoding and storage.
    ```python
    # Simplified example of the workaround
    value_base64 = base64.b64encode(value_bytes).decode("ascii")
    transaction.execute_update(
        "INSERT ... VALUES (@key, FROM_BASE64(@val))",
        params={"key": key, "val": value_base64},
        param_types={"val": spanner.param_types.STRING}
    )
    ```
*   **Read Path**: While `FROM_BASE64` should store raw bytes, we retained a heuristic Base64 decoding check in `get()` as a defensive measure against potential double-encoding scenarios observed during development.

### 3. Native Table Writes
We refactored `graphrag.utils.storage.write_table_to_storage` (and related load functions) to check if the storage backend supports a native `set_table` method. `SpannerPipelineStorage` implements this to write DataFrames directly to Spanner tables (mapping DataFrame columns to Spanner columns) instead of serializing them to Parquet files.

### 4. Automatic Spanner Table Creation
Both `SpannerPipelineStorage` (for general data tables) and `SpannerVectorStore` (for vector embeddings) now support **automatic table creation**.
*   **Mechanism**: When a write operation (`set_table` or `load_documents`) encounters a `NotFound` error indicating the table is missing, it automatically infers the schema from the data (or configuration for vectors) and executes the necessary DDL to create the table before retrying the write.
*   **Vector Index**: For `SpannerVectorStore`, it also automatically creates a default **Vector Index** (using COSINE distance) to ensure high-performance ANN search out-of-the-box.
*   **Benefit**: Significantly simplifies deployment, as users do not need to manually execute DDL scripts before running the pipeline.