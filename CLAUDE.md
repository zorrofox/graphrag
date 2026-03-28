## Project Context

This is a fork of [microsoft/graphrag](https://github.com/microsoft/graphrag) (`main` branch), extended to support Google Cloud Platform (GCP) resources. Upstream changes should be merged from `microsoft/graphrag:main` periodically.

**Custom additions** (not in upstream):
- GCS pipeline storage backend
- Cloud Spanner pipeline storage + vector store backend
- GCP-specific config fields in `StorageConfig` and `VectorStoreConfig`

## Commands

```bash
uv sync                       # Install dependencies (Python 3.10–3.12 required)
uv run poe index <args>       # Run indexing pipeline
uv run poe query <args>       # Run query engine
uv run poe prompt_tune <args> # Run prompt tuning
uv run poe test               # Run all tests
uv run poe test_unit          # Unit tests only
uv run poe test_integration   # Integration tests only
uv run poe test_smoke         # Smoke tests only
uv run poe check              # Lint + type-check + security + format check
uv run poe fix                # Auto-fix formatting
./scripts/start-azurite.sh    # Start Azurite emulator (required for some storage tests)
```

**Before submitting a PR:** `uv run semversioner add-change -t patch -d "<description>."`

## Architecture

Factory pattern throughout — each subsystem has a `factory.py` entrypoint for swapping implementations.

```
graphrag/
├── api/             # Public library API
├── cache/           # Cache backends (factory.py)
├── cli/             # CLI entrypoints (main.py)
├── config/          # Pydantic config models + loaders
│   └── models/      # StorageConfig, VectorStoreConfig, etc.
├── index/           # Indexing pipeline (run/run.py is main entrypoint)
├── language_model/  # LLM wrappers (fnllm-based)
├── prompt_tune/     # Prompt tuning module
├── prompts/         # System prompts
├── query/           # Query engine
├── storage/         # Storage backends (factory.py)
├── vector_stores/   # Vector store backends (factory.py)
└── utils/           # Shared helpers (storage.py has set_table/load_table hooks)
```

## GCP Adaptation (Custom)

This fork extends GraphRAG with native Google Cloud Platform support. Key files added:

| File | Purpose |
|------|---------|
| `graphrag/storage/gcs_pipeline_storage.py` | GCS-backed PipelineStorage (blob ops) |
| `graphrag/storage/spanner_pipeline_storage.py` | Spanner-backed PipelineStorage (native DataFrame writes via `set_table`) |
| `graphrag/vector_stores/spanner.py` | Spanner vector store (COSINE_DISTANCE, auto-creates table + vector index) |
| `tests/unit/storage/test_gcs_pipeline_storage.py` | GCS unit tests |
| `tests/unit/storage/test_spanner_pipeline_storage.py` | Spanner storage unit tests |
| `tests/unit/vector_stores/test_spanner.py` | Spanner vector store unit tests |
| `tests/integration/storage/test_gcp_integration.py` | GCP integration tests (skipped by default) |

**Config model changes** (`graphrag/config/models/`):
- `StorageConfig`: added `bucket_name`, `project_id`, `instance_id`, `database_id`, `table_prefix`
- `VectorStoreConfig`: added `project_id`, `instance_id`, `database_id`

## Testing

### Unit tests

```bash
uv run poe test_unit                     # all unit tests
uv run python -m pytest tests/unit/storage/test_gcs_pipeline_storage.py \
                           tests/unit/storage/test_spanner_pipeline_storage.py \
                           tests/unit/vector_stores/test_spanner.py \
                           tests/unit/utils/test_spanner_resource_manager.py -v
```

All GCP unit tests use `unittest.mock` — no live credentials needed.

| Test file | What it covers |
|-----------|---------------|
| `tests/unit/storage/test_gcs_pipeline_storage.py` | GCSPipelineStorage CRUD, child(), find(), keys(), clear(), async exception handling |
| `tests/unit/storage/test_spanner_pipeline_storage.py` | SpannerPipelineStorage blob ops, set_table() auto-create/alter/retry, `TestInferSpannerType` (13 type-inference cases), `TestSafeIdentifier` (SQL injection guard) |
| `tests/unit/vector_stores/test_spanner.py` | SpannerVectorStore load, search, filter, SQL identifier quoting, query_filter lifecycle |
| `tests/unit/utils/test_spanner_resource_manager.py` | SpannerResourceManager client sharing, credentials-aware keying, reference counting |

### Integration tests

Require live GCP credentials. Set env vars before running:

```bash
export GRAPHRAG_GCP_INTEGRATION_TEST=1
export GCS_BUCKET_NAME=<bucket>
export GCP_PROJECT_ID=<project>
export SPANNER_INSTANCE_ID=<instance>
export SPANNER_DATABASE_ID=<database>

uv run poe test_integration
# or target only GCP tests:
uv run python -m pytest tests/integration/storage/test_gcp_integration.py -v
```

**Spanner instance requirements:** Enterprise edition, `GOOGLE_STANDARD_SQL` dialect (needed for `ARRAY<FLOAT64>(vector_length=>N)` and vector indexes).

```bash
# Create a temporary test instance (us-central1, Enterprise, 100 PUs)
gcloud spanner instances create <name> \
  --project=<project> --config=regional-us-central1 \
  --edition=ENTERPRISE --processing-units=100
gcloud spanner databases create <db> --instance=<name> \
  --database-dialect=GOOGLE_STANDARD_SQL
```

| Test | What it covers |
|------|---------------|
| `test_gcs_storage_integration` | GCS set / has / get / delete |
| `test_gcs_child_and_find` | child() client sharing, find() with pattern + file_filter |
| `test_gcs_keys_and_clear` | keys() listing, clear() |
| `test_gcs_cache_factory_integration` | GCS-backed JsonPipelineCache via CacheFactory |
| `test_spanner_storage_integration` | Spanner blob CRUD |
| `test_spanner_child_prefix_isolation` | child() prefix stacking, parent/child table isolation |
| `test_spanner_find_and_keys` | find() pattern + file_filter + max_count, keys() |
| `test_spanner_table_storage_integration` | DataFrame write/read with JSON / ARRAY / null |
| `test_spanner_auto_table_creation` | set_table() auto-creates missing table |
| `test_spanner_schema_evolution` | set_table() auto-alters table to add columns |
| `test_spanner_load_empty_table_columns` | load_table() on empty table returns correct columns |
| `test_spanner_vector_store_integration` | Vector load, search_by_id, similarity search |
| `test_spanner_vector_store_auto_creation` | Vector table + index auto-created on first write |
| `test_spanner_vector_auto_create_with_length` | Verifies `vector_length=>N` constraint in INFORMATION_SCHEMA |
| `test_spanner_storage_factory_integration` | StorageFactory creates SpannerPipelineStorage from config |

## Gotchas

- **Spanner BYTES encoding**: The Python client inconsistently handles `bytes` columns. Always Base64-encode writes explicitly and use `FROM_BASE64()` in DML. Retain heuristic Base64 decoding on reads as a defensive measure.
- **Native table writes**: `write_table_to_storage` checks for a `set_table` hook on the backend — `SpannerPipelineStorage` implements this to bypass Parquet serialization entirely.
- **Auto table creation**: Both `SpannerPipelineStorage` and `SpannerVectorStore` auto-create tables (and vector index) on first write when a `NotFound` error is encountered — no manual DDL needed.
- **Azurite required**: Some unit/smoke tests fail without Azurite running.
- **Between minor version bumps**: Run `graphrag init --root [path] --force` to get the latest config format (overwrites config + prompts — backup first).
- **Unit tests use mocks**: GCP unit tests use `unittest.mock` — no live credentials needed. Integration tests require real GCP credentials and are skipped by default.
