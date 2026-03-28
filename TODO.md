# GCP Adaptation — Outstanding Work

Items that have not yet been implemented, grouped by priority.
Completed items are tracked in git history.

---

## Performance

### ~~P1 — Spanner I/O in async methods blocks the event loop~~ ✅ Done

Fixed in commit `fed67ee`. Every async method now delegates blocking Spanner
calls to a private `_*_sync` counterpart via `asyncio.to_thread()`:
`load_table`, `has_table`, `get`, `has`, `delete`, `clear`, `get_creation_date`,
`set_table` (`_batch_insert`), and both `run_in_transaction` calls in `set()`.

---

### ~~P2 — GCS `find()` and `keys()` block the calling thread~~ ✅ Done

Mitigated in commit `c385d6f`. `list_blobs()` now receives
`max_results=1000, page_size=1000` in both `find()` and `keys()` to cap
page-descriptor size and reduce per-call latency.  Full async requires a
native async GCS client (tracked separately below).

---

### ~~P3 — No retry / exponential back-off for transient GCS errors~~ ✅ Done

Fixed in commit `c385d6f`. Every `asyncio.to_thread(...)` call in
`get()`, `set()`, `has()`, `delete()`, `clear()`, and `get_creation_date()` is
now wrapped with `_GCS_RETRY = api_retry.AsyncRetry(...)` (initial 1 s,
multiplier 2×, max 60 s, deadline 300 s) that retries HTTP 429 / 500 / 503.

---

### ~~P4 — `_schema_cache` is unbounded~~ ✅ Done

Fixed in commit `fed67ee`. The cache is now a `collections.OrderedDict`
capped at `_SCHEMA_CACHE_MAX_SIZE = 256` entries; the oldest entry is evicted
(LRU) when the limit is reached.

---

## Correctness / Semantic Bugs

### ~~P1 — `_infer_spanner_type()` samples only the first non-null value~~ ✅ Done

Fixed in commit `fed67ee`. The object-dtype branch now samples up to 10
non-null values, classifies each independently, and takes a majority vote.
If the vote is not unanimous the function falls back to `"JSON"`.  `bool`
is now checked before `str`/`int` to avoid mis-classifying Python bools.

---

### ~~P2 — `load_documents(overwrite=True)` does not truncate the table~~ ✅ Done

Fixed in commit `642d045`. `overwrite=True` now issues
`execute_partitioned_dml("DELETE FROM ...")` before inserting, matching
LanceDB behaviour. Auto-creation path correctly skips the DELETE when the
table does not yet exist.

---

### ~~P3 — `get_creation_date()` returns last-modified time, not creation time~~ ✅ Done

Fixed in commit `fed67ee`.
- `_ensure_blobs_table_exists()` DDL now includes both `created_at` and
  `updated_at` columns (`allow_commit_timestamp=true`).
- `set()` tries `INSERT INTO` first (sets `created_at`); on `AlreadyExists`
  falls back to `UPDATE` (preserves `created_at`, updates `updated_at` only).
- `get_creation_date()` reads the `created_at` column.

Note: existing Spanner databases need a manual `ALTER TABLE` to add the
`created_at` column before the new behaviour takes effect.

---

### ~~P4 — `load_table()` loads entire table into memory~~ ✅ Done

Fixed in commit `fed67ee`. `load_table()` (and its sync twin) now accept
optional `limit: int | None` and `offset: int = 0` parameters that append
`LIMIT @lim OFFSET @off` to the SQL query.  `load_table_from_storage()` in
`utils/storage.py` passes them through.

---

## New Features

### ~~Spanner Emulator support~~ ✅ Done

Fixed in commit `14a7cd7`. `_build_spanner_client()` in
`SpannerResourceManager` detects `SPANNER_EMULATOR_HOST` and automatically
uses `AnonymousCredentials` — no real GCP credentials needed for local dev.

---

### ~~Vertex AI Vector Search backend~~ ✅ Done

Fixed in commit `fed67ee`. `graphrag/vector_stores/vertexai.py` implements
`VertexAIVectorStore` using the `google-cloud-aiplatform` SDK
(`MatchingEngineIndex` / `MatchingEngineIndexEndpoint`).
`VectorStoreType.VertexAI = "vertexai"` added to enums; registered in
factory; config fields (`location`, `index_id`, `index_endpoint_id`,
`deployed_index_id`) added to `VectorStoreConfig`.
13/13 unit tests passing.

---

### ~~LiteLLM response cache backed by GCS~~ ✅ Done

Fixed in commit `604f209`. `graphrag/cache/gcs_litellm_cache.py` implements
`GCSLiteLLMCache(BaseCache)` wrapping `GCSPipelineStorage`.
Registered as `CacheType.gcs_litellm` in the cache factory.
10/10 unit tests passing.

---

### GCS async client upgrade

Replace `google-cloud-storage` (synchronous) with a native async GCS client
so `asyncio.to_thread()` is no longer needed.

**Dependency change:** add `google-cloud-storage[async]` to `pyproject.toml`.
Deferred — the current `asyncio.to_thread()` approach is correct and the
async SDK is not yet production-stable for all operations.

---

*Last updated: 2026-03-28 — all items complete except GCS async client (deferred)*
