## Project Context

This is a fork of [microsoft/graphrag](https://github.com/microsoft/graphrag) (`feature/v3-migration` branch), ported from the old monolithic v2.7.x structure to the **v3.0.8 monorepo**. Upstream base tag: `v3.0.8`.

**Custom additions** (not in upstream):
- GCS storage backend (`graphrag-storage` package)
- Cloud Spanner storage + vector store backends
- Vertex AI Vector Search backend (`graphrag-vectors` package)
- GCS LiteLLM response cache (`graphrag-cache` package)
- GCP-specific config fields in `StorageConfig` and `VectorStoreConfig`

## Commands

```bash
uv sync                       # Install / refresh workspace dependencies (Python 3.11–3.13)
uv run poe test_unit          # Unit tests only (no credentials needed)
uv run poe test_integration   # Integration tests (requires live GCP creds)
uv run poe check              # Lint (ruff) + type-check (pyright) + format check
uv run poe fix                # Auto-fix formatting
```

**Before submitting a PR:**
```bash
uv run semversioner add-change -t minor -d "feat: GCP backends for v3 monorepo"
```

## Architecture

### Monorepo layout (v3)

```
packages/
├── graphrag/            # Main package — CLI, config, index workflows, query engine
├── graphrag-storage/    # Storage backends  (graphrag_storage.*)
├── graphrag-vectors/    # Vector store backends (graphrag_vectors.*)
├── graphrag-cache/      # Cache backends (graphrag_cache.*)
├── graphrag-common/     # Base Factory class, ServiceScope
├── graphrag-chunking/   # Text chunking
├── graphrag-input/      # Input readers
└── graphrag-llm/        # LLM wrappers
tests/
├── unit/
│   ├── storage/         # Storage unit tests (incl. GCS)
│   ├── vector_stores/   # Vector store unit tests (incl. Spanner)
│   └── utils/           # Utility unit tests (incl. SpannerResourceManager)
└── integration/
    └── storage/         # GCP integration tests
```

### Factory pattern

Each package exposes `create_*` + `register_*` functions:

```python
from graphrag_storage import create_storage, StorageConfig
from graphrag_vectors import create_vector_store, VectorStoreConfig
from graphrag_cache import create_cache, CacheConfig
```

## GCP Adaptation (Custom)

### New files added by this fork

| Package | File | Purpose |
|---------|------|---------|
| `graphrag-storage` | `graphrag_storage/gcs_storage.py` | GCS-backed `Storage` (blob CRUD, find, keys, clear) |
| `graphrag-storage` | `graphrag_storage/spanner_storage.py` | Spanner-backed `Storage` (blob ops + `set_table`/`load_table`/`has_table`) |
| `graphrag-storage` | `graphrag_storage/spanner_resource_manager.py` | Shared Spanner client lifecycle (ref-counted, credentials-aware) |
| `graphrag-vectors` | `graphrag_vectors/spanner.py` | Spanner vector store (COSINE_DISTANCE, v3 `VectorStore` ABC) |
| `graphrag-vectors` | `graphrag_vectors/vertexai.py` | Vertex AI Vector Search backend |
| `graphrag-cache` | `graphrag_cache/gcs_litellm_cache.py` | GCS-backed LiteLLM cache + graphrag `Cache` ABC |

### Modified upstream files

| File | Change |
|------|--------|
| `graphrag_storage/storage_type.py` | Added `GCS = "gcs"`, `Spanner = "spanner"` |
| `graphrag_storage/storage_config.py` | Added `bucket_name`, `project_id`, `instance_id`, `database_id`, `table_prefix` |
| `graphrag_storage/storage_factory.py` | Lazy-load registration for GCS + Spanner |
| `graphrag_vectors/vector_store_type.py` | Added `Spanner = "spanner"`, `VertexAI = "vertexai"` |
| `graphrag_vectors/vector_store_config.py` | Added Spanner + Vertex AI connection fields |
| `graphrag_vectors/vector_store_factory.py` | Lazy-load registration for Spanner + VertexAI |
| `graphrag_cache/cache_type.py` | Added `GCSLiteLLM = "gcs_litellm"` |
| `graphrag_cache/cache_factory.py` | Lazy-load registration for GCSLiteLLM |
| `packages/graphrag-storage/pyproject.toml` | Added `google-cloud-storage`, `google-cloud-spanner` |
| `packages/graphrag-vectors/pyproject.toml` | Added `google-cloud-spanner`, `google-cloud-aiplatform` |
| `packages/graphrag-cache/pyproject.toml` | Added `google-cloud-storage`, `litellm` |

### v3 interface changes vs v2

| v2 (monolith) | v3 (monorepo) |
|---|---|
| `from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage` | `from graphrag_storage.gcs_storage import GCSStorage` |
| `from graphrag.storage.spanner_pipeline_storage import SpannerPipelineStorage` | `from graphrag_storage.spanner_storage import SpannerStorage` |
| `from graphrag.vector_stores.spanner import SpannerVectorStore` | `from graphrag_vectors.spanner import SpannerVectorStore` |
| `from graphrag.vector_stores.vertexai import VertexAIVectorStore` | `from graphrag_vectors.vertexai import VertexAIVectorStore` |
| `from graphrag.cache.gcs_litellm_cache import GCSLiteLLMCache` | `from graphrag_cache.gcs_litellm_cache import GCSLiteLLMCache` |
| `from graphrag.utils.spanner_resource_manager import SpannerResourceManager` | `from graphrag_storage.spanner_resource_manager import SpannerResourceManager` |
| `VectorStoreDocument(id=.., text=.., vector=.., attributes=..)` | `VectorStoreDocument(id=.., vector=.., data={"text": .., ..})` |
| `store.connect(**kwargs)` | `store.connect()` (kwargs passed to `__init__`) |
| `load_documents(docs, overwrite=True)` | `load_documents(docs)` (always overwrites) |
| `retrieved.text`, `retrieved.attributes` | `retrieved.data.get("text")`, `retrieved.data.get("attributes")` |
| `find()` yields `(name, group_dict)` tuples | `find()` yields `str` (key name only) |

## Testing

### Unit tests (no credentials needed)

```bash
uv run poe test_unit

# GCP-specific only:
uv run python -m pytest \
  tests/unit/storage/test_gcs_storage.py \
  tests/unit/utils/test_spanner_resource_manager.py \
  tests/unit/vector_stores/test_spanner.py -v
```

| Test file | What it covers |
|-----------|---------------|
| `tests/unit/storage/test_gcs_storage.py` | GCSStorage CRUD, child(), find(), keys(), clear(), retry, async exceptions |
| `tests/unit/utils/test_spanner_resource_manager.py` | Client sharing, credentials-aware keying, ref-counting, emulator support |
| `tests/unit/vector_stores/test_spanner.py` | SpannerVectorStore load, search_by_id, similarity search, count, remove, update, FilterExpr, select |

### Integration tests

Require live GCP credentials:

```bash
export GRAPHRAG_GCP_INTEGRATION_TEST=1
export GCS_BUCKET_NAME=<bucket>
export GCP_PROJECT_ID=<project>
export SPANNER_INSTANCE_ID=<instance>
export SPANNER_DATABASE_ID=<database>

uv run python -m pytest tests/integration/storage/test_gcp_integration.py -v
```

**Spanner instance requirements:** Enterprise edition, `GOOGLE_STANDARD_SQL` dialect.

```bash
# Temporary test instance
gcloud spanner instances create <name> \
  --project=<project> --config=regional-us-central1 \
  --edition=ENTERPRISE --processing-units=100
gcloud spanner databases create <db> --instance=<name> \
  --database-dialect=GOOGLE_STANDARD_SQL
```

| Test | What it covers |
|------|---------------|
| `test_gcs_storage_integration` | GCS set / has / get / delete |
| `test_gcs_child_and_find` | child() client sharing, find() pattern matching |
| `test_gcs_keys_and_clear` | keys() listing, clear() |
| `test_gcs_litellm_cache_integration` | GCS LiteLLM cache async get/set |
| `test_spanner_storage_integration` | Spanner blob CRUD |
| `test_spanner_child_prefix_isolation` | child() prefix stacking, parent/child table isolation |
| `test_spanner_find_and_keys` | find() pattern matching, keys() |
| `test_spanner_table_storage_integration` | DataFrame write/read (JSON / ARRAY / null) |
| `test_spanner_auto_table_creation` | set_table() auto-creates missing table |
| `test_spanner_schema_evolution` | set_table() auto-alters table to add columns |
| `test_spanner_load_empty_table_columns` | load_table() on empty table returns correct columns |
| `test_spanner_load_table_pagination` | load_table(limit=N, offset=M) pagination |
| `test_spanner_blob_creation_date_preserved_on_update` | created_at preserved across updates |
| `test_spanner_vector_store_integration` | Vector load, search_by_id, similarity search, count |
| `test_spanner_vector_store_auto_creation` | Vector table + index auto-created on first write |
| `test_spanner_vector_auto_create_with_length` | Verifies `vector_length=>N` in INFORMATION_SCHEMA |
| `test_spanner_storage_factory_integration` | create_storage() produces SpannerStorage from config |

## Cloud Run Deployment

End-to-end deployment scripts and configuration live in `deploy/`. See `docs/gcp_integration.md` section 13 for the full guide.

### Resource Naming Convention

| Resource | Recommended name pattern |
|----------|--------------------------|
| Cloud Run Service | `graphrag-query-service` |
| Cloud Run Job | `graphrag-indexer` |
| GCS buckets | `<project>-graphrag-{input,index,cache}` |
| Spanner instance | `graphrag-instance` (Enterprise, min 100 PU) |
| Spanner database | `graphrag-db` (GOOGLE_STANDARD_SQL dialect) |
| Artifact Registry | `<region>-docker.pkg.dev/<project>/graphrag` |
| Service accounts | `graphrag-query-sa`, `graphrag-indexer-sa` |

### Configuration Choices

- **LLM**: `gemini-3-flash-preview` via Vertex AI with `VERTEXAI_LOCATION=global`
- **Embedding**: `text-embedding-005` via Vertex AI (768 dims)
- **Auth**: `auth_method: azure_managed_identity` — bypasses the `api_key` requirement; LiteLLM's `vertex_ai` provider uses ADC (Cloud Run Workload Identity) automatically
- **Cache**: `type: memory` — `GCSLiteLLMCache` is not usable here (see gotchas below)
- **Vector store**: Spanner with auto-DDL; tables and vector indexes are created on first write

### Deployment Quick Reference

```bash
# Run from the repository root

# One-time infrastructure setup
bash deploy/infra/01_setup_gcp.sh

# Build and push Docker images
bash deploy/infra/02_build_push.sh

# Create indexer job and run full indexing
bash deploy/infra/03_deploy_jobs.sh run

# Deploy query service
bash deploy/infra/04_deploy_query.sh

# Enable IAP and grant user access
bash deploy/infra/05_setup_iap.sh
bash deploy/infra/05_setup_iap.sh grant user@your-domain.com

# Trigger incremental update
bash deploy/infra/03_deploy_jobs.sh update

# Test query (from within GCP environment)
SERVICE_URL="https://<your-cloud-run-url>"
ID_TOKEN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${SERVICE_URL}&format=full" -H "Metadata-Flavor: Google")
curl -X POST "$SERVICE_URL/v1/query/global" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?"}'
```

### Cloud Run Deployment Gotchas

- **`uv sync` requires `--all-packages`**: Without this flag, workspace member packages (graphrag, graphrag-storage, etc.) are not installed in the container.
- **Escape `$` in `settings.yaml` regex**: `settings.yaml` is processed by `string.Template.substitute()`. A literal `$` in values like `file_pattern` must be written as `$$` (e.g. `".*\\.txt$$"`).
- **Do not use uvloop**: GraphRAG uses `nest_asyncio2` internally, which cannot patch uvloop's event loop. Use plain `uvicorn` (not `uvicorn[standard]`) and do not pass `--loop uvloop`.
- **`GCSLiteLLMCache` is not usable as the GraphRAG cache**: The factory hashes all LLM constructor arguments using `yaml.dump()` to build a singleton cache key. `GCSLiteLLMCache` holds a `google.cloud.storage.Client` which raises `PicklingError` during serialization and crashes `extract_graph`. Use `cache.type: memory` instead.
- **`PipelineRunResult.error`** (not `.errors`): The attribute name is singular.
- **Spanner built-in metrics disabled in code**: `_build_spanner_client()` passes `disable_builtin_metrics=True` to every `spanner.Client()`. Without this, the SDK creates an `OtelPeriodicExportingMetricReader` thread and registers a `MeterProvider.shutdown(timeout=30 s)` atexit handler; when the Cloud Monitoring export fails (400 InvalidArgument — incomplete resource labels), that handler blocks process exit for 30 seconds. Note: the env var `SPANNER_ENABLE_BUILT_IN_METRICS` is **not** read by the SDK — the correct env var would be `SPANNER_DISABLE_BUILTIN_METRICS=true`, but the in-code flag is more reliable.
- **`gemini-3-flash-preview` requires `VERTEXAI_LOCATION=global`**: This model is only available on the global endpoint; setting the location to a specific region (e.g. `us-central1`) returns 404.
- **Single uvicorn worker**: `SpannerResourceManager` uses a module-level singleton that is not multiprocess-safe. Cloud Run must run with `--workers 1`; scale horizontally via additional instances.
- **IAP binding command**: Grant `roles/iap.httpsResourceAccessor` via `gcloud iap web add-iam-policy-binding`, not `gcloud run services add-iam-policy-binding` (the latter does not support this role on Cloud Run resources).

## Gotchas

- **VectorStoreDocument**: v3 uses `data: dict[str, Any]` instead of separate `text` and `attributes` fields. Store text as `data["text"]` and retrieve with `doc.data.get("text")`.
- **VectorStore new required methods**: v3 ABC adds `count()`, `remove(ids)`, `update(document)` — all implemented in `SpannerVectorStore`. `FilterExpr` filters are applied in-memory post-query.
- **Spanner BYTES encoding**: Always Base64-encode writes with `FROM_BASE64()` in DML. Retain heuristic Base64 decoding on reads for backward compatibility.
- **SpannerStorage.set_table / load_table / has_table**: Extra methods beyond the `Storage` ABC — used by workflows that need native DataFrame writes without Parquet round-trips.
- **Auto table creation**: Both `SpannerStorage` and `SpannerVectorStore` auto-create tables and vector indexes on first write — no manual DDL needed.
- **SpannerResourceManager location**: Moved from `graphrag.utils.spanner_resource_manager` to `graphrag_storage.spanner_resource_manager` in v3.
- **Unit tests use mocks**: GCP unit tests use `unittest.mock` — no live credentials needed.
- **Spanner Emulator**: Set `SPANNER_EMULATOR_HOST=localhost:9010` to redirect to the local emulator. `AnonymousCredentials` are used automatically.
- **Spanner metrics noise**: Resolved — `_build_spanner_client()` now passes `disable_builtin_metrics=True`, preventing the `OtelPeriodicExportingMetricReader` thread and its atexit handler from being created entirely.
- **ruff pre-existing errors**: `uv run poe check` reports ~170 ruff errors that existed in v3.0.8 before our changes. Pyright reports 0 errors from our code.
