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

### GCS async client upgrade — 调研结论：暂缓，当前方案已是最佳实践

**调研日期：** 2026-03-28

#### 现状

| 方案 | 状态 | 说明 |
|------|------|------|
| `google-cloud-storage` 官方 async gRPC 客户端 | ❌ 实验性 | v3.4.0 (2024-09) 引入，代码在 `_experimental/` 模块，未覆盖标准 REST 上传/下载路径 |
| `gcloud-aio-storage` | ✅ 生产就绪 | 社区维护，基于 `aiohttp` 实现真正 async I/O，v7.x 稳定，但引入新依赖且 API 与现有接口差异较大 |
| `gcsfs` | ✅ 部分可用 | `asynchronous=True` 支持，但 `open()` 仍同步 |

#### 结论

当前 `asyncio.to_thread()` 方案是正确的，原因：
1. 官方 `google-cloud-storage` async 支持尚在 `_experimental/`，不建议生产依赖
2. `gcloud-aio-storage` 迁移成本高（API 不兼容，需重写 `GCSPipelineStorage`）
3. `asyncio.to_thread()` 将阻塞 I/O 移入线程池，对 GCS 这类网络 I/O 操作实际开销极小

#### 触发升级的条件

满足以下任意一条时再重新评估：
- 官方 `google-cloud-storage` async 支持升级为稳定版（移出 `_experimental/`）
- 性能分析（profiling）证明线程池成为瓶颈
- 项目决定引入 `gcloud-aio-storage` 作为新依赖

**参考：**
- [Issue #1366 — Async Storage Client](https://github.com/googleapis/python-storage/issues/1366)
- [PR #1537 — feat(experimental): add async grpc client](https://github.com/googleapis/python-storage/pull/1537)
- [gcloud-aio-storage](https://github.com/talkiq/gcloud-aio)

---

*Last updated: 2026-03-28 — 所有 TODO 项已完成；GCS async upgrade 调研后决定暂缓*
