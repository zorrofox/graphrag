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

## Gotchas

- **Spanner BYTES encoding**: The Python client inconsistently handles `bytes` columns. Always Base64-encode writes explicitly and use `FROM_BASE64()` in DML. Retain heuristic Base64 decoding on reads as a defensive measure.
- **Native table writes**: `write_table_to_storage` checks for a `set_table` hook on the backend — `SpannerPipelineStorage` implements this to bypass Parquet serialization entirely.
- **Auto table creation**: Both `SpannerPipelineStorage` and `SpannerVectorStore` auto-create tables (and vector index) on first write when a `NotFound` error is encountered — no manual DDL needed.
- **Azurite required**: Some unit/smoke tests fail without Azurite running.
- **Between minor version bumps**: Run `graphrag init --root [path] --force` to get the latest config format (overwrites config + prompts — backup first).
- **Unit tests use mocks**: GCP unit tests use `unittest.mock` — no live credentials needed. Integration tests require real GCP credentials and are skipped by default.
