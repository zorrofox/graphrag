# GCP Adaptation — Outstanding Work

Items that have not yet been implemented, grouped by priority.
Completed items are tracked in git history.

---

## Performance

### P1 — Spanner I/O in async methods still blocks the event loop

**Files:** `spanner_pipeline_storage.py`, `spanner_vector_store.py`

DDL calls were moved to `asyncio.to_thread()` but all ordinary Spanner reads
and writes remain synchronous inside `async` methods:

| Method | Blocking call |
|--------|--------------|
| `load_table()` | `snapshot.execute_sql()` |
| `has_table()` | `snapshot.execute_sql()` |
| `get()` | `snapshot.read()` |
| `has()` | `snapshot.read()` |
| `delete()` | `batch.delete()` |
| `set_table()` | `_batch_insert()` → `database.batch()` |
| `keys()` / `find()` | `snapshot.execute_sql()` |
| `get_creation_date()` | `snapshot.read()` |

**Suggested fix:** extract a synchronous twin (`_set_table_sync`, etc.) for
each method and have the `async` wrapper call it with `asyncio.to_thread()`.

---

### P2 — GCS `find()` and `keys()` block the calling thread

**File:** `gcs_pipeline_storage.py`

Both methods are synchronous (base-class contract) but call
`self._client.list_blobs()` which does network I/O.  They cannot be made
`async` without changing the `PipelineStorage` interface.

**Suggested fix:** use the `google-cloud-storage-async` package, or batch the
listing calls with the `max_results` parameter to reduce per-call latency.

---

### P3 — No retry / exponential back-off for transient GCS errors

**File:** `gcs_pipeline_storage.py`

HTTP 429 / 503 responses raise immediately.  Other GCS backends (Azure Blob)
have built-in retry policies.

**Suggested fix:** wrap every `asyncio.to_thread(...)` call in a small retry
helper with exponential back-off (e.g. `tenacity` library, already available
transitively, or `google-api-core`'s `retry` module).

---

### P4 — `_schema_cache` is unbounded

**File:** `spanner_pipeline_storage.py`

The in-memory schema cache grows without limit for long-running pipelines that
touch many tables.

**Suggested fix:** use `functools.lru_cache` or a bounded `OrderedDict` (e.g.
max 256 entries, evict LRU).

---

## Correctness / Semantic Bugs

### P1 — `_infer_spanner_type()` samples only the first non-null value

**File:** `spanner_pipeline_storage.py` — `_infer_spanner_type()`

For object-dtype columns the function inspects `non_null.iloc[0]`.  A column
whose first row happens to be an empty list (inferred as
`ARRAY<STRING(MAX)>`) but later rows contain dicts would be mis-typed.

**Suggested fix:** scan the first *N* non-null values (e.g. 10) and take the
most common type, or raise a `ValueError` on conflicting types.

---

### ~~P2 — `load_documents(overwrite=True)` does not truncate the table~~ ✅ Done

Fixed in commit `642d045`. `overwrite=True` now issues
`execute_partitioned_dml("DELETE FROM ...")` before inserting, matching
LanceDB behaviour. Auto-creation path correctly skips the DELETE when the
table does not yet exist.

---

### P3 — `get_creation_date()` returns last-modified time, not creation time

**File:** `spanner_pipeline_storage.py` — `get_creation_date()` / blob table DDL

The blob table stores a single `updated_at TIMESTAMP` column populated by
`PENDING_COMMIT_TIMESTAMP()` on every write, so it reflects the last
modification, not the original creation.

**Suggested fix:**
1. Add a `created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)`
   column to the blob table DDL in `_ensure_blobs_table_exists()`.
2. Change the DML in `set()` to set `created_at` only on `INSERT` (use Spanner
   `INSERT` mutation + handle `AlreadyExists` with a separate `UPDATE`
   mutation, or use a `MERGE`-style approach via read-before-write).
3. Read `created_at` in `get_creation_date()`.

Note: this requires a schema migration for existing Spanner databases.

---

### P4 — `load_table()` loads entire table into memory

**File:** `spanner_pipeline_storage.py` — `load_table()`

There is no pagination; a million-row `entities` table will exhaust RAM.

**Suggested fix:** add an optional `limit` / `offset` parameter, or use
Spanner partition queries (`database.run_partition_query()`) for true
server-side streaming.  The caller in `graphrag/utils/storage.py` would need
a corresponding `load_table_in_batches()` helper.

---

## New Features

### ~~Spanner Emulator support~~ ✅ Done

Implemented in `feat: Spanner Emulator support via SPANNER_EMULATOR_HOST env var` — `_build_spanner_client()` helper in `SpannerResourceManager` detects `SPANNER_EMULATOR_HOST` and automatically uses `AnonymousCredentials`.

---

### ~~Vertex AI Vector Search backend~~ DONE

~~Implement a `VertexAIVectorStore(BaseVectorStore)` for billion-scale workloads.~~

**Implemented:** `graphrag/vector_stores/vertexai.py` — `VertexAIVectorStore` using
`google-cloud-aiplatform` SDK (`MatchingEngineIndex` / `MatchingEngineIndexEndpoint`).
`VectorStoreType.VertexAI = "vertexai"` added to enums; registered in factory;
config fields (`location`, `index_id`, `index_endpoint_id`, `deployed_index_id`)
added to `VectorStoreConfig`.

---

### ~~LiteLLM response cache backed by GCS~~ DONE

~~The project already depends on `litellm`.  Register `GCSPipelineStorage` as a
LiteLLM cache backend so repeated LLM calls with identical prompts are served
from GCS instead of hitting the API.~~

**Implemented:** `graphrag/cache/gcs_litellm_cache.py` — `GCSLiteLLMCache` wraps
`GCSPipelineStorage` and satisfies the LiteLLM `BaseCache` interface.
Registered as `CacheType.gcs_litellm` in the cache factory.
Tests in `tests/unit/cache/test_gcs_litellm_cache.py` (10/10 passing).

---

### GCS async client upgrade

Replace `google-cloud-storage` (synchronous) with
`google-cloud-storage-async` or use the `AsyncClient` from the same package.
This would let `get()`, `set()`, `has()`, `delete()`, `clear()`, and
`get_creation_date()` perform true async I/O without `asyncio.to_thread()`.

**Dependency change:** add `google-cloud-storage[async]` to `pyproject.toml`.

---

*Last updated: 2026-03-28*
