# GCP Integration

This project supports Google Cloud Platform (GCP) services for storage and vector search.

## Prerequisites

- A GCP project with billing enabled.
- Enabled APIs:
    - Cloud Storage API
    - Cloud Spanner API
- A Google Cloud Storage bucket.
- A Google Cloud Spanner instance and database.

## Configuration

GraphRAG supports using environment variables in `settings.yaml` via the `${VAR_NAME}` syntax. This is the recommended way to manage GCP credentials and resource IDs.

### 1. Define Environment Variables

Create a `.env` file in your project root (or set these variables in your environment):

```bash
# Common GCP settings
GRAPHRAG_GCP_PROJECT_ID=my-gcp-project-id
GRAPHRAG_SPANNER_INSTANCE_ID=my-spanner-instance
GRAPHRAG_SPANNER_DATABASE_ID=graphrag-db

# Storage specific
GRAPHRAG_INPUT_BUCKET_NAME=my-corp-data-bucket
GRAPHRAG_CACHE_BUCKET_NAME=my-corp-cache-bucket
```

### 2. Configure `settings.yaml`

Use the environment variables in your `settings.yaml` file.

```yaml
input:
  type: gcs
  bucket_name: ${GRAPHRAG_INPUT_BUCKET_NAME}
  base_dir: inputs/financial_reports_q1

storage:
  # Use Spanner for structured output storage
  type: spanner
  project_id: ${GRAPHRAG_GCP_PROJECT_ID}
  instance_id: ${GRAPHRAG_SPANNER_INSTANCE_ID}
  database_id: ${GRAPHRAG_SPANNER_DATABASE_ID}
  # table_prefix: "optional_prefix_"

cache:
  # Use GCS for caching
  type: gcs
  bucket_name: ${GRAPHRAG_CACHE_BUCKET_NAME}
  base_dir: graphrag_cache

embeddings:
  async_mode: threaded
  vector_store:
    # Use Spanner for vector storage
    type: spanner
    project_id: ${GRAPHRAG_GCP_PROJECT_ID}
    instance_id: ${GRAPHRAG_SPANNER_INSTANCE_ID}
    database_id: ${GRAPHRAG_SPANNER_DATABASE_ID}
  target:
    # Ensure these tables exist in Spanner (see DDL below)
    entity_description: entity_description_embeddings
    community_description: community_description_embeddings
```

## Spanner Schema

### Automatic Table Creation

The Spanner integration supports **automatic table creation**. When the pipeline runs, if it attempts to write to a table that does not exist (e.g., `documents`, `create_final_nodes`, or the `Blobs` table), it will automatically infer the schema from the data and create the table in Spanner.

**Note:** Automatic creation might take a few seconds per table as Spanner executes the DDL.

### Vector Store Tables

Vector store tables are **automatically created** if they do not exist when the pipeline first attempts to write embeddings.

*   **Schema**: Inferred from your configuration (including `vector_size`).
*   **Index**: A default **Vector Index** is automatically created for high-performance approximate nearest neighbor (ANN) search.
    *   Uses `COSINE` distance.
    *   Indexes only non-null vectors (`WHERE vector IS NOT NULL`).

By default, GraphRAG often uses the following collection names:
*   `entity_description_embeddings`
*   `community_description_embeddings`

#### Manual Creation (Advanced)

If you need to customize the vector index (e.g., change distance type, add index options), you can still manually create the table and index before running the pipeline.

```sql
-- 1. Create the table
CREATE TABLE YourCollectionName (
    id STRING(MAX) NOT NULL,
    vector ARRAY<FLOAT64>(vector_length=>1536), -- Must specify length
    text STRING(MAX),
    attributes JSON,
) PRIMARY KEY (id);

-- 2. Create a custom vector index
CREATE VECTOR INDEX YourCollectionIndex ON YourCollectionName(vector)
WHERE vector IS NOT NULL
OPTIONS (distance_type = 'EUCLIDEAN_SQUARED'); -- Example customization
```

## Developer Notes

- **Spanner `BYTES` Handling:** The `google-cloud-spanner` Python client may have issues automatically encoding `bytes` objects for `BYTES` columns in some scenarios (e.g., `batch.insert_or_update`). The `SpannerPipelineStorage` implementation uses a workaround involving explicit Base64 encoding and the `FROM_BASE64` SQL function in DML statements for writes, and a heuristic Base64 decoding for reads if necessary.
- **Automatic Schema Inference:** The `SpannerPipelineStorage` infers Spanner types from Pandas DataFrames. It defaults to `STRING(MAX)` for unknown object types and `JSON` for complex nested structures (lists of dicts, etc.).
