# GraphRAG GCP Integration Guide

本文档描述如何在 GraphRAG v3 monorepo 中使用 Google Cloud Platform (GCP) 后端，包括架构设计、资源创建、配置方法和运维建议。

---

## 目录

1. [架构概览](#1-架构概览)
2. [GCP 后端组件](#2-gcp-后端组件)
   - [GCS 存储 (GCSStorage)](#21-gcs-存储-gcsstorage)
   - [Spanner 存储 (SpannerStorage)](#22-spanner-存储-spannerstorage)
   - [Spanner 向量存储 (SpannerVectorStore)](#23-spanner-向量存储-spannervectorstore)
   - [Vertex AI 向量存储 (VertexAIVectorStore)](#24-vertex-ai-向量存储-vertexaivectorstore)
   - [GCS LiteLLM 缓存 (GCSLiteLLMCache)](#25-gcs-litellm-缓存-gcslitellmcache)
   - [Spanner 资源管理器 (SpannerResourceManager)](#26-spanner-资源管理器-spannerresourcemanager)
3. [前置条件](#3-前置条件)
4. [GCP 资源创建](#4-gcp-资源创建)
   - [GCS Bucket](#41-gcs-bucket)
   - [Spanner 实例与数据库](#42-spanner-实例与数据库)
   - [Vertex AI Vector Search 索引](#43-vertex-ai-vector-search-索引)
5. [身份验证](#5-身份验证)
6. [Python 配置示例](#6-python-配置示例)
7. [Spanner 表结构说明](#7-spanner-表结构说明)
   - [Blob 存储表](#71-blob-存储表)
   - [DataFrame 存储表](#72-dataframe-存储表)
   - [向量存储表](#73-向量存储表)
8. [运行集成测试](#8-运行集成测试)
9. [Spanner Emulator（本地开发）](#9-spanner-emulator本地开发)
10. [IAM 权限参考](#10-iam-权限参考)
11. [费用与规格建议](#11-费用与规格建议)
12. [常见问题](#12-常见问题)
13. [Cloud Run 生产部署](#13-cloud-run-生产部署)
    - [架构概览](#131-架构概览)
    - [目录结构](#132-目录结构)
    - [settings.yaml 配置](#133-settingsyaml-配置)
    - [分步部署](#134-分步部署)
    - [IAP 访问控制](#135-iap-访问控制)
    - [查询 API 端点](#136-查询-api-端点)
    - [已知限制与注意事项](#137-已知限制与注意事项)

---

## 1. 架构概览

GraphRAG v3 采用 monorepo 结构，GCP 支持以插件形式分布在三个包中：

```
packages/
├── graphrag-storage/
│   ├── graphrag_storage/gcs_storage.py          ← GCS blob 存储
│   ├── graphrag_storage/spanner_storage.py      ← Spanner blob + DataFrame 存储
│   └── graphrag_storage/spanner_resource_manager.py  ← 共享连接池
├── graphrag-vectors/
│   ├── graphrag_vectors/spanner.py              ← Spanner 向量存储
│   └── graphrag_vectors/vertexai.py             ← Vertex AI 向量搜索
└── graphrag-cache/
    └── graphrag_cache/gcs_litellm_cache.py      ← GCS LiteLLM 响应缓存
```

各组件均通过工厂函数注册，使用方不感知底层实现：

```
GraphRAG 管道
     │
     ├── create_storage(StorageConfig(type="gcs"))       → GCSStorage
     ├── create_storage(StorageConfig(type="spanner"))   → SpannerStorage
     ├── create_vector_store(VectorStoreConfig(type="spanner"))  → SpannerVectorStore
     ├── create_vector_store(VectorStoreConfig(type="vertexai")) → VertexAIVectorStore
     └── create_cache(CacheConfig(type="gcs_litellm"))   → GCSLiteLLMCache
```

**数据流示意（以 Spanner 全链路为例）：**

```
索引阶段
  文档 → 分块 → 嵌入 → SpannerVectorStore（向量 + JSON data）
  中间文件 → SpannerStorage（Blob 表 / DataFrame 表）

查询阶段
  查询向量 → SpannerVectorStore.similarity_search_by_vector()
           → COSINE_DISTANCE 排序 → 返回 VectorStoreSearchResult

LLM 调用
  请求 → GCSLiteLLMCache（命中则返回） → LiteLLM → 响应写入 GCS
```

---

## 2. GCP 后端组件

### 2.1 GCS 存储 (GCSStorage)

**包**：`graphrag-storage`
**类**：`graphrag_storage.gcs_storage.GCSStorage`
**用途**：将 GraphRAG 的索引产物（JSON、Parquet 等文件）存储到 Google Cloud Storage。

**特点**：
- 所有写/读操作异步（`asyncio.to_thread` + Google API 重试）
- `child(name)` 共享底层 GCS Client，避免每个子目录开新 TCP 连接
- 内置指数退避重试：`TooManyRequests`、`ServiceUnavailable`、`InternalServerError`

**配置字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `"gcs"` | 固定值 |
| `bucket_name` | `str` | GCS bucket 名称 |
| `base_dir` | `str` | bucket 内的路径前缀（可选） |
| `encoding` | `str` | 文本编码，默认 `utf-8` |

---

### 2.2 Spanner 存储 (SpannerStorage)

**包**：`graphrag-storage`
**类**：`graphrag_storage.spanner_storage.SpannerStorage`
**用途**：将 GraphRAG 的中间文件存储到 Spanner，同时支持 DataFrame 的原生写入（绕过 Parquet 序列化）。

**超出 `Storage` ABC 的额外方法**：

| 方法 | 说明 |
|------|------|
| `set_table(name, df)` | 将 DataFrame 写入 Spanner 表，自动建表或补列 |
| `load_table(name, limit, offset)` | 分页读取 Spanner 表，返回 DataFrame |
| `has_table(name)` | 检查指定表是否存在 |

**自动 DDL**：
- `set_table()` 首次写入时自动推断列类型并 `CREATE TABLE`
- 若后续写入包含新列，自动执行 `ALTER TABLE ADD COLUMN`
- 不需要手动执行任何 DDL

**配置字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `"spanner"` | 固定值 |
| `project_id` | `str` | GCP 项目 ID |
| `instance_id` | `str` | Spanner 实例 ID |
| `database_id` | `str` | Spanner 数据库 ID |
| `table_prefix` | `str` | 表名前缀，用于命名空间隔离（可选） |

---

### 2.3 Spanner 向量存储 (SpannerVectorStore)

**包**：`graphrag-vectors`
**类**：`graphrag_vectors.spanner.SpannerVectorStore`
**用途**：将文档向量存储到 Spanner，使用 `COSINE_DISTANCE` + 原生 Vector Index 进行 ANN 搜索。

**表结构（每个索引对应一张表）**：

```sql
CREATE TABLE `<index_name>` (
    `id`          STRING(MAX) NOT NULL,
    `vector`      ARRAY<FLOAT64>(vector_length=><N>),
    `data`        JSON,          -- 存储 text、attributes 及所有元数据
    `create_date` STRING(MAX),
    `update_date` STRING(MAX)
) PRIMARY KEY (`id`)
```

**自动 DDL**：`load_documents()` 首次调用时若表不存在，自动创建表 + Vector Index。

**FilterExpr**：`similarity_search_by_vector()` 支持 `filters` 参数，过滤逻辑在 Python 内存中执行（Spanner 侧先返回 TOP-K，再按 filter 筛选）。

**配置字段**：与 SpannerStorage 相同（`project_id`、`instance_id`、`database_id`）。

---

### 2.4 Vertex AI 向量存储 (VertexAIVectorStore)

**包**：`graphrag-vectors`
**类**：`graphrag_vectors.vertexai.VertexAIVectorStore`
**用途**：使用 Vertex AI Vector Search（MatchingEngine）进行大规模 ANN 搜索。

**限制**：
- Index 和 IndexEndpoint **必须预先创建**（本组件不负责创建）
- 不支持 `search_by_id()`（返回 stub）和 `count()`（返回 0）
- `upsert_datapoints()` 为异步操作，新插入向量不立即可搜

**配置字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `"vertexai"` | 固定值 |
| `project_id` | `str` | GCP 项目 ID |
| `location` | `str` | GCP 区域，如 `us-central1` |
| `index_id` | `str` | MatchingEngineIndex 完整资源名或数字 ID |
| `index_endpoint_id` | `str` | MatchingEngineIndexEndpoint 完整资源名或数字 ID |
| `deployed_index_id` | `str` | Endpoint 上已部署的 Index ID |

---

### 2.5 GCS LiteLLM 缓存 (GCSLiteLLMCache)

**包**：`graphrag-cache`
**类**：`graphrag_cache.gcs_litellm_cache.GCSLiteLLMCache`
**用途**：将 LLM 响应持久化到 GCS，实现跨进程/跨运行的 LLM 调用缓存。

**双重接口**：
- 实现 graphrag `Cache` ABC（`get`/`set`/`has`/`delete`/`clear`/`child`）
- 实现 LiteLLM `BaseCache` 接口（`async_get_cache`/`async_set_cache`），可直接接入 LiteLLM

**接入 LiteLLM**：

```python
import litellm
from graphrag_cache.gcs_litellm_cache import GCSLiteLLMCache
from litellm import Cache

cache = GCSLiteLLMCache(bucket_name="my-bucket", base_dir="llm-cache")
litellm.cache = Cache(type="custom", cache_instance=cache)
```

**配置字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `"gcs_litellm"` | 固定值（通过工厂创建时） |
| `bucket_name` | `str` | GCS bucket 名称 |
| `base_dir` | `str` | GCS 路径前缀，默认 `litellm-cache` |

---

### 2.6 Spanner 资源管理器 (SpannerResourceManager)

**包**：`graphrag-storage`
**类**：`graphrag_storage.spanner_resource_manager.SpannerResourceManager`

全局单例，确保同一 `(project, credentials, instance, database)` 组合只创建一个 `spanner.Client` 和一个 `spanner.Database` 对象，避免多个 `SpannerStorage`/`SpannerVectorStore` 实例各自维护独立连接池导致线程泄漏。

- **引用计数**：每次 `get_database()` +1，`release_database()` -1，引用归零时关闭 Client
- **Credentials-aware**：不同 service account 使用不同 Client（通过 `service_account_email` 区分）
- **Emulator 支持**：检测 `SPANNER_EMULATOR_HOST`，自动使用 `AnonymousCredentials`

---

## 3. 前置条件

### 软件依赖

```bash
# Python 3.11+
uv sync  # 自动安装所有依赖
```

依赖已声明在各包的 `pyproject.toml` 中，无需手动安装：

| 包 | GCP 依赖 |
|----|---------|
| `graphrag-storage` | `google-cloud-storage>=2.10`, `google-cloud-spanner>=3.40` |
| `graphrag-vectors` | `google-cloud-spanner>=3.40`, `google-cloud-aiplatform>=1.60` |
| `graphrag-cache` | `google-cloud-storage>=2.10`, `litellm>=1.0` |

### gcloud CLI

```bash
# 安装 Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# 验证安装
gcloud --version
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
```

---

## 4. GCP 资源创建

### 4.1 GCS Bucket

```bash
# 创建 bucket（选择距离计算资源最近的区域）
gcloud storage buckets create gs://<BUCKET_NAME> \
  --project=<PROJECT_ID> \
  --location=us-central1 \
  --uniform-bucket-level-access

# 验证
gcloud storage buckets describe gs://<BUCKET_NAME>
```

**命名建议**：
- 格式：`<project-id>-graphrag-<环境>`，如 `myproject-graphrag-prod`
- bucket 名称全局唯一，小写字母、数字、连字符

---

### 4.2 Spanner 实例与数据库

#### 创建实例

GraphRAG 使用 Spanner Vector Index，**必须** 使用 Enterprise 或 Enterprise Plus 版本。

```bash
# 生产环境（Enterprise，按需调整 processing-units）
gcloud spanner instances create <INSTANCE_ID> \
  --project=<PROJECT_ID> \
  --config=regional-us-central1 \
  --edition=ENTERPRISE \
  --processing-units=1000 \
  --description="GraphRAG production"

# 开发/测试环境（最低规格，100 PU ≈ 0.1 node）
gcloud spanner instances create <INSTANCE_ID> \
  --project=<PROJECT_ID> \
  --config=regional-us-central1 \
  --edition=ENTERPRISE \
  --processing-units=100 \
  --description="GraphRAG dev"
```

**实例配置参考**：

| 场景 | 版本 | Processing Units | 说明 |
|------|------|-----------------|------|
| 开发/测试 | Enterprise | 100 | 最低规格，不保证 SLA |
| 小规模生产 | Enterprise | 1000 (1 node) | 支持 Vector Index |
| 中规模生产 | Enterprise Plus | 3000+ | 更高吞吐和 SLA |

> **注意**：Enterprise Plus 支持更大的 vector_length（>= 某些限制），Enterprise 已足够支持常见嵌入维度（768、1536、3072）。

#### 创建数据库

```bash
# 必须使用 GOOGLE_STANDARD_SQL（PostgreSQL 方言不支持 Vector Index）
gcloud spanner databases create <DATABASE_ID> \
  --instance=<INSTANCE_ID> \
  --project=<PROJECT_ID> \
  --database-dialect=GOOGLE_STANDARD_SQL

# 验证
gcloud spanner databases describe <DATABASE_ID> \
  --instance=<INSTANCE_ID> \
  --project=<PROJECT_ID>
```

#### DDL 说明

GraphRAG 会**自动创建**所有需要的表，无需手动执行 DDL。但如需提前创建（例如在 CI/CD 中预置），可参考以下模板：

```sql
-- Blob 存储表（SpannerStorage 用）
CREATE TABLE `<prefix>Blobs` (
    `key`        STRING(MAX) NOT NULL,
    `value`      BYTES(MAX),
    `created_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true),
    `updated_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (`key`);

-- 向量存储表（SpannerVectorStore 用，N 为嵌入维度）
CREATE TABLE `<index_name>` (
    `id`          STRING(MAX) NOT NULL,
    `vector`      ARRAY<FLOAT64>(vector_length=><N>),
    `data`        JSON,
    `create_date` STRING(MAX),
    `update_date` STRING(MAX)
) PRIMARY KEY (`id`);

CREATE VECTOR INDEX `<index_name>_VectorIndex`
    ON `<index_name>`(`vector`)
    WHERE `vector` IS NOT NULL
    OPTIONS (distance_type = 'COSINE');
```

> **注意**：Vector Index 创建需要 5–15 分钟，请耐心等待。

---

### 4.3 Vertex AI Vector Search 索引

Vertex AI Vector Search 的 Index 和 IndexEndpoint 必须在使用前预先创建。

#### 步骤一：创建 Index

```bash
# 通过 gcloud（推荐用 Python SDK 或 Console 创建，gcloud 支持有限）
# 以下为使用 Python SDK 的示例

python3 - <<'EOF'
from google.cloud import aiplatform

aiplatform.init(project="<PROJECT_ID>", location="us-central1")

# 创建 TreeAhNN Index（适合大规模数据集）
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="graphrag-index",
    dimensions=768,              # 与嵌入模型维度一致
    approximate_neighbors_count=150,
    distance_measure_type="COSINE_DISTANCE",
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=7,
    description="GraphRAG vector index",
)
print(f"Index resource name: {index.resource_name}")
EOF
```

#### 步骤二：创建 IndexEndpoint

```bash
python3 - <<'EOF'
from google.cloud import aiplatform

aiplatform.init(project="<PROJECT_ID>", location="us-central1")

endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="graphrag-endpoint",
    public_endpoint_enabled=True,
    description="GraphRAG vector search endpoint",
)
print(f"Endpoint resource name: {endpoint.resource_name}")
EOF
```

#### 步骤三：部署 Index 到 Endpoint

```bash
python3 - <<'EOF'
from google.cloud import aiplatform

aiplatform.init(project="<PROJECT_ID>", location="us-central1")

endpoint = aiplatform.MatchingEngineIndexEndpoint("<ENDPOINT_RESOURCE_NAME>")
endpoint.deploy_index(
    index=aiplatform.MatchingEngineIndex("<INDEX_RESOURCE_NAME>"),
    deployed_index_id="graphrag_deployed",   # 自定义，作为 deployed_index_id 参数
    display_name="graphrag-deployed-index",
    min_replica_count=1,
    max_replica_count=2,
)
print("Index deployed successfully")
EOF
```

#### 获取配置值

```bash
# 列出所有 Index
gcloud ai indexes list --project=<PROJECT_ID> --region=us-central1

# 列出所有 IndexEndpoint
gcloud ai index-endpoints list --project=<PROJECT_ID> --region=us-central1

# 查看 Endpoint 上已部署的 Index（获取 deployed_index_id）
gcloud ai index-endpoints describe <ENDPOINT_ID> \
  --project=<PROJECT_ID> --region=us-central1 \
  --format="json(deployedIndexes)"
```

---

## 5. 身份验证

### 本地开发（Application Default Credentials）

```bash
gcloud auth application-default login
```

ADC 会被 GCS、Spanner、Vertex AI SDK 自动检测，无需任何代码配置。

### GKE / Compute Engine（Workload Identity / 默认 SA）

在 GKE 上推荐使用 Workload Identity，Pod 的 Kubernetes Service Account 绑定 GCP Service Account，无需在代码或环境变量中管理密钥：

```bash
# 绑定 Workload Identity
gcloud iam service-accounts add-iam-policy-binding \
  graphrag-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:<PROJECT_ID>.svc.id.goog[<NAMESPACE>/<KSA_NAME>]"
```

### Service Account Key（不推荐，仅作为备选）

```bash
# 创建 SA
gcloud iam service-accounts create graphrag-sa \
  --project=<PROJECT_ID> \
  --display-name="GraphRAG Service Account"

# 生成密钥
gcloud iam service-accounts keys create ./sa-key.json \
  --iam-account=graphrag-sa@<PROJECT_ID>.iam.gserviceaccount.com

# 使用
export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json
```

---

## 6. Python 配置示例

### GCS 存储

```python
from graphrag_storage import create_storage, StorageConfig
from graphrag_storage.storage_type import StorageType

config = StorageConfig(
    type=StorageType.GCS,
    bucket_name="myproject-graphrag-prod",
    base_dir="indexing-output",
)
storage = create_storage(config)

# 写入
await storage.set("entities.json", '{"entities": [...]}')

# 读取
data = await storage.get("entities.json")

# 子目录（共享 GCS Client，不开新连接）
child = storage.child("community-reports")
await child.set("report-1.json", "...")
```

### Spanner 存储

```python
from graphrag_storage import create_storage, StorageConfig
from graphrag_storage.storage_type import StorageType

config = StorageConfig(
    type=StorageType.Spanner,
    project_id="myproject",
    instance_id="graphrag-prod",
    database_id="graphrag-db",
    table_prefix="Prod_",   # 可选，用于同库多环境隔离
)
storage = create_storage(config)

# Blob 操作
await storage.set("pipeline-state.json", b'{"status": "running"}')
data = await storage.get("pipeline-state.json", as_bytes=True)

# DataFrame 写入（绕过 Parquet，直接写 Spanner）
import pandas as pd
df = pd.DataFrame([
    {"id": "e1", "name": "Alice", "type": "person"},
    {"id": "e2", "name": "Acme Corp", "type": "organization"},
])
await storage.set_table("Entities", df)   # 自动建表

# DataFrame 读取（支持分页）
all_entities = await storage.load_table("Entities")
page = await storage.load_table("Entities", limit=100, offset=0)
```

### Spanner 向量存储

```python
from graphrag_vectors import create_vector_store, VectorStoreConfig
from graphrag_vectors.index_schema import IndexSchema
from graphrag_vectors.vector_store import VectorStoreDocument
from graphrag_vectors.vector_store_type import VectorStoreType
from graphrag_vectors.filtering import F

config = VectorStoreConfig(
    type=VectorStoreType.Spanner,
    project_id="myproject",
    instance_id="graphrag-prod",
    database_id="graphrag-db",
    vector_size=768,
)
schema = IndexSchema(index_name="entity_embeddings", vector_size=768)

store = create_vector_store(config, schema)
store.connect()

# 写入（自动建表 + Vector Index）
docs = [
    VectorStoreDocument(
        id="e1",
        vector=[0.1, 0.2, ...],               # 768 维
        data={"text": "Alice", "type": "person"},
    ),
]
store.load_documents(docs)

# 相似度搜索
query_vector = embed("Who is the CEO?")
results = store.similarity_search_by_vector(query_vector, k=10)
for r in results:
    print(r.document.id, r.score, r.document.data.get("text"))

# 带过滤的搜索
results = store.similarity_search_by_vector(
    query_vector,
    k=10,
    filters=(F.type == "person") & (F.community_id == "c1"),
)

# 查询总数
print(store.count())

store.close()
```

### Vertex AI 向量存储

```python
from graphrag_vectors import create_vector_store, VectorStoreConfig
from graphrag_vectors.index_schema import IndexSchema
from graphrag_vectors.vector_store_type import VectorStoreType

config = VectorStoreConfig(
    type=VectorStoreType.VertexAI,
    project_id="myproject",
    location="us-central1",
    index_id="projects/123/locations/us-central1/indexes/456",
    index_endpoint_id="projects/123/locations/us-central1/indexEndpoints/789",
    deployed_index_id="graphrag_deployed",
    vector_size=768,
)
schema = IndexSchema(index_name="vertexai_index", vector_size=768)

store = create_vector_store(config, schema)
store.connect()

results = store.similarity_search_by_vector(query_vector, k=10)
store.close()
```

### GCS LiteLLM 缓存

```python
# 方式一：作为 graphrag Cache 使用
from graphrag_cache import create_cache, CacheConfig
from graphrag_cache.cache_type import CacheType

config = CacheConfig(type=CacheType.GCSLiteLLM)
# 通过 StorageConfig 嵌套传入 GCS 配置
# （或直接实例化）

# 方式二：直接实例化并接入 LiteLLM
import litellm
from graphrag_cache.gcs_litellm_cache import GCSLiteLLMCache
from litellm import Cache

cache = GCSLiteLLMCache(
    bucket_name="myproject-graphrag-prod",
    base_dir="llm-response-cache",
)
litellm.cache = Cache(type="custom", cache_instance=cache)

# 后续所有 litellm.completion() 调用自动命中缓存
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## 7. Spanner 表结构说明

### 7.1 Blob 存储表

每个 `SpannerStorage` 实例操作一张 Blob 表（默认名为 `<prefix>Blobs`）：

```sql
CREATE TABLE `<prefix>Blobs` (
    `key`        STRING(MAX) NOT NULL,
    `value`      BYTES(MAX),
    `created_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true),
    `updated_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (`key`);
```

- `value` 以 Base64 编码存储（历史兼容性）
- `created_at` 首次写入时设置，后续 update 不修改（通过双事务保证）

### 7.2 DataFrame 存储表

`set_table()` 会根据 DataFrame 的列动态推断并创建表：

| pandas dtype | Spanner 类型 |
|--------------|-------------|
| int64 | `INT64` |
| float64 | `FLOAT64` |
| bool | `BOOL` |
| datetime64 | `TIMESTAMP` |
| object (str) | `STRING(MAX)` |
| object (dict) | `JSON` |
| object (list[str]) | `ARRAY<STRING(MAX)>` |
| object (list[int]) | `ARRAY<INT64>` |
| object (list[float]) | `ARRAY<FLOAT64>` |
| object (混合) | `JSON` |

### 7.3 向量存储表

每个 `SpannerVectorStore` 对应一张向量表（`index_name` 即表名）：

```sql
CREATE TABLE `<index_name>` (
    `id`          STRING(MAX) NOT NULL,
    `vector`      ARRAY<FLOAT64>(vector_length=><N>),  -- N 为嵌入维度
    `data`        JSON,    -- 存储所有非向量字段（text、metadata 等）
    `create_date` STRING(MAX),
    `update_date` STRING(MAX)
) PRIMARY KEY (`id`);

CREATE VECTOR INDEX `<index_name>_VectorIndex`
    ON `<index_name>`(`vector`)
    WHERE `vector` IS NOT NULL
    OPTIONS (distance_type = 'COSINE');
```

- `data` 列以 JSON 存储所有 `VectorStoreDocument.data` 字段
- `create_date` / `update_date` 为 ISO 8601 字符串，同时在 `data` 中存储展开的时间分量（`year`、`month`、`quarter` 等）供 FilterExpr 过滤

---

## 8. 运行集成测试

### 环境变量

```bash
export GRAPHRAG_GCP_INTEGRATION_TEST=1
export GCS_BUCKET_NAME=<bucket>
export GCP_PROJECT_ID=<project>
export SPANNER_INSTANCE_ID=<instance>
export SPANNER_DATABASE_ID=<database>

# Vertex AI（可选）
export VERTEXAI_LOCATION=us-central1
export VERTEXAI_INDEX_ID=projects/.../indexes/...
export VERTEXAI_INDEX_ENDPOINT_ID=projects/.../indexEndpoints/...
export VERTEXAI_DEPLOYED_INDEX_ID=<deployed-id>
export VERTEXAI_VECTOR_SIZE=768
```

### 创建临时测试资源

```bash
# Bucket
gcloud storage buckets create gs://${GCS_BUCKET_NAME} \
  --project=${GCP_PROJECT_ID} --location=us-central1 \
  --uniform-bucket-level-access

# Spanner（100 PU，Enterprise，测试用最低配）
gcloud spanner instances create ${SPANNER_INSTANCE_ID} \
  --project=${GCP_PROJECT_ID} \
  --config=regional-us-central1 \
  --edition=ENTERPRISE \
  --processing-units=100 \
  --description="GraphRAG test"

gcloud spanner databases create ${SPANNER_DATABASE_ID} \
  --instance=${SPANNER_INSTANCE_ID} \
  --project=${GCP_PROJECT_ID} \
  --database-dialect=GOOGLE_STANDARD_SQL
```

### 运行测试

```bash
# 存储集成测试（GCS + Spanner 存储）
uv run python -m pytest tests/integration/storage/test_gcp_integration.py -v

# Spanner 向量存储集成测试
uv run python -m pytest tests/integration/vector_stores/test_spanner.py -v

# Vertex AI 向量存储集成测试（需要预建 Index）
uv run python -m pytest tests/integration/vector_stores/test_vertexai.py -v

# 全部 GCP 集成测试
uv run python -m pytest \
  tests/integration/storage/test_gcp_integration.py \
  tests/integration/vector_stores/test_spanner.py \
  -v
```

> **注意**：`test_spanner.py` 首次运行时需要 5–15 分钟创建 Spanner Vector Index（module-scoped fixture，整个测试模块只创建一次）。

### 清理测试资源

```bash
gcloud spanner databases delete ${SPANNER_DATABASE_ID} \
  --instance=${SPANNER_INSTANCE_ID} --project=${GCP_PROJECT_ID} --quiet

gcloud spanner instances delete ${SPANNER_INSTANCE_ID} \
  --project=${GCP_PROJECT_ID} --quiet

gcloud storage rm -r gs://${GCS_BUCKET_NAME} --quiet
```

---

## 9. Spanner Emulator（本地开发）

使用 Spanner Emulator 可在本地无 GCP 凭据运行开发和单元测试：

```bash
# 启动 Emulator（Docker）
docker run -d -p 9010:9010 -p 9020:9020 \
  gcr.io/cloud-spanner-emulator/emulator

# 配置环境变量（自动使用 AnonymousCredentials）
export SPANNER_EMULATOR_HOST=localhost:9010

# 创建实例和数据库（Emulator 不需要真实 GCP 项目）
gcloud spanner instances create test-instance \
  --project=test-project \
  --config=emulator-config \
  --description="Local test" \
  --nodes=1

gcloud spanner databases create test-db \
  --instance=test-instance \
  --project=test-project
```

> **注意**：Spanner Emulator 不支持 Vector Index DDL，因此 `SpannerVectorStore` 的 `create_index()` 会失败。Emulator 适合测试 `SpannerStorage` 的 Blob 操作和 DataFrame 读写。

---

## 10. IAM 权限参考

### GCS

```bash
gcloud storage buckets add-iam-policy-binding gs://<BUCKET> \
  --member="serviceAccount:<SA_EMAIL>" \
  --role="roles/storage.objectAdmin"
```

最小权限：`roles/storage.objectUser`（读写对象，无删除）或 `roles/storage.objectAdmin`（含删除）。

### Cloud Spanner

```bash
gcloud spanner databases add-iam-policy-binding <DATABASE_ID> \
  --instance=<INSTANCE_ID> \
  --project=<PROJECT_ID> \
  --member="serviceAccount:<SA_EMAIL>" \
  --role="roles/spanner.databaseUser"
```

| 角色 | 说明 |
|------|------|
| `roles/spanner.databaseUser` | 执行 DML（INSERT/UPDATE/DELETE/SELECT） |
| `roles/spanner.databaseAdmin` | DML + DDL（CREATE/ALTER TABLE），GraphRAG 自动建表需要此权限 |

> **推荐**：生产环境中分两个 SA：DDL SA（`databaseAdmin`）仅用于初始化，运行时 SA 使用 `databaseUser`。

### Vertex AI

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:<SA_EMAIL>" \
  --role="roles/aiplatform.user"
```

---

## 11. 费用与规格建议

### Spanner

| 规格 | 费用（us-central1，约） | 适用场景 |
|------|------------------------|---------|
| 100 PU（Enterprise） | ~$65/月 | 开发/测试 |
| 1000 PU = 1 node | ~$650/月 | 小规模生产 |
| 3000 PU（Enterprise Plus） | ~$2,100+/月 | 中规模生产 |

> Vector Index 在 Enterprise 版本可用，是使用 Spanner 向量搜索的最低要求。

### GCS

- 存储费用约 $0.020/GB/月（Standard 存储类，us-central1）
- GraphRAG 索引产物通常在 1–10 GB 量级，存储费用可忽略
- 主要成本来自 API 请求次数（Class A/B 操作）

### Vertex AI Vector Search

- Index 存储：约 $0.34/GB/月
- 查询：按 QPS 和 replica 数计费，具体参考官方定价
- **重要**：Index 创建（30–60 分钟）和 replica 按小时计费，测试后及时删除

---

## 12. 常见问题

### Spanner Vector Index 创建慢

Vector Index DDL 需要 5–15 分钟属于正常。`create_index()` 内部使用 `operation.result(timeout=900)` 等待完成。可通过 GCP Console 或以下命令查看进度：

```bash
gcloud spanner operations list \
  --instance=<INSTANCE_ID> \
  --project=<PROJECT_ID> \
  --filter="metadata.@type:UpdateDatabaseDdlMetadata"
```

### Spanner OTEL 指标噪声

测试输出中出现大量 `InvalidArgument: One or more TimeSeries could not be written` 是 Spanner Python 客户端发送 OpenTelemetry 指标时的已知问题（时间序列发送过于频繁），**不影响功能**，可忽略。

### 向量维度不匹配

`ARRAY<FLOAT64>(vector_length=>N)` 一旦建立就不能更改维度。若需更换嵌入模型，必须 `DROP TABLE` 重建。建议在 `index_name` 中包含模型信息，如 `entity_embeddings_text3_768`。

### Spanner Blob 写入报 `AlreadyExists`

`SpannerStorage.set()` 对已存在的 key 使用 UPDATE 而非 INSERT，应自动处理。若仍出现此错误，通常是并发写入同一 key，检查调用方是否存在竞争条件。

### Vertex AI 搜索结果未包含刚插入的向量

Vertex AI `upsert_datapoints()` 为异步操作，新写入的向量通常在 30 秒至数分钟后才可搜索。在集成测试中可通过 `VERTEXAI_SEARCH_WAIT_SECONDS` 环境变量控制等待时间。

### 本地 ADC 权限不足

```bash
# 确认当前身份
gcloud auth list
gcloud auth application-default print-access-token | python3 -c "
import sys, json, urllib.request
token = sys.stdin.read().strip()
req = urllib.request.Request('https://oauth2.googleapis.com/tokeninfo?access_token=' + token)
print(json.loads(urllib.request.urlopen(req).read())['email'])
"

# 重新登录 ADC
gcloud auth application-default login
```

---

## 13. Cloud Run 生产部署

本节描述将 GraphRAG v3 部署到 GCP Cloud Run 的完整方案：
- **Cloud Run Service**：对外提供 HTTP 查询接口（global / local / drift / basic 四种搜索）
- **Cloud Run Job**：以批处理方式运行索引构建与增量更新

所有脚本和配置位于仓库 `deploy/` 目录下，所有脚本均从**仓库根目录**执行。

### 13.1 架构概览

```
Client（IAP 认证）
    │  HTTPS
    ▼
Cloud Run Service: graphrag-query-service
    │  FastAPI + graphrag.api
    │  启动时从 GCS 加载 parquet DataFrames 到内存
    │
    ├── GCS (<project>-graphrag-index)   ← 读取 *.parquet 索引文件
    ├── Spanner (graphrag-db)            ← 向量相似度搜索（per-request）
    └── Vertex AI Gemini                 ← LLM 补全

Cloud Run Job: graphrag-indexer
    ├── GCS (<project>-graphrag-input)   ← 读取原始文档
    ├── GCS (<project>-graphrag-index)   ← 写入 parquet 索引文件
    └── Spanner (graphrag-db)            ← 写入向量表（auto-DDL）
```

### 13.2 目录结构

```
deploy/
├── config/
│   └── settings.yaml          # GraphRAG 配置模板（${ENV_VAR} 占位符）
├── query-service/
│   ├── app/
│   │   ├── main.py            # FastAPI 应用（4 个搜索端点 + healthz/readyz）
│   │   └── loader.py          # 启动加载器：GCS parquet → 内存 DataFrames
│   ├── Dockerfile
│   └── requirements.txt       # fastapi + uvicorn（不含 uvloop，见已知限制）
├── indexer/
│   ├── entrypoint.py          # 调用 api.build_index()，支持增量更新
│   └── Dockerfile
└── infra/
    ├── 01_setup_gcp.sh        # 一次性基础设施：GCS、Spanner、SA、IAM、Artifact Registry
    ├── 02_build_push.sh       # Docker build + push 到 Artifact Registry
    ├── 03_deploy_jobs.sh      # 创建/更新 Cloud Run Job；触发全量或增量运行
    ├── 04_deploy_query.sh     # 部署 Cloud Run Service
    └── 05_setup_iap.sh        # 启用 IAP；为用户/用户组授权
```

### 13.3 settings.yaml 配置

`deploy/config/settings.yaml` 是 GraphRAG 的统一配置，所有敏感值通过 `${ENV_VAR}` 占位符在运行时注入（Python `string.Template`）。

**LLM 配置（Vertex AI Gemini，无需 API key）：**

```yaml
completion_models:
  default_completion_model:
    model_provider: vertex_ai
    model: gemini-3-flash-preview   # 需要 VERTEXAI_LOCATION=global
    auth_method: azure_managed_identity  # 跳过 api_key 校验，使用 ADC/Workload Identity
    call_args:
      temperature: 0
      max_tokens: 4096

embedding_models:
  default_embedding_model:
    model_provider: vertex_ai
    model: text-embedding-005
    auth_method: azure_managed_identity
```

> `auth_method: azure_managed_identity` 绕过 `api_key` 必填校验。LiteLLM 的 `vertex_ai` provider 自动回退到 Application Default Credentials（ADC），由 Cloud Run Workload Identity 自动提供，无需密钥文件。

**存储、表提供者、向量存储与缓存：**

```yaml
output_storage:
  type: gcs
  bucket_name: ${GCS_BUCKET_INDEX}
  base_dir: output

table_provider:
  type: parquet

vector_store:
  type: spanner
  project_id: ${GRAPHRAG_PROJECT_ID}
  instance_id: ${SPANNER_INSTANCE_ID}
  database_id: ${SPANNER_DATABASE_ID}
  vector_size: 768

cache:
  type: memory   # GCSLiteLLMCache 在此场景不可用，见「已知限制」
```

**运行时环境变量：**

| 变量 | 说明 |
|------|------|
| `GRAPHRAG_PROJECT_ID` | GCP 项目 ID |
| `GCS_BUCKET_INPUT` | 输入文档 bucket |
| `GCS_BUCKET_INDEX` | 索引输出 bucket（parquet 文件） |
| `GCS_BUCKET_CACHE` | LLM 响应缓存 bucket（预留，暂不启用） |
| `SPANNER_INSTANCE_ID` | Spanner 实例 ID |
| `SPANNER_DATABASE_ID` | Spanner 数据库 ID |
| `GOOGLE_CLOUD_PROJECT` | GCP 项目 ID（供 google-cloud SDK 使用） |
| `VERTEXAI_PROJECT` | GCP 项目 ID（供 LiteLLM vertex_ai provider 使用） |
| `VERTEXAI_LOCATION` | `global`——`gemini-3-flash-preview` 仅在 global endpoint 可用 |
| `SPANNER_ENABLE_BUILT_IN_METRICS` | 设为 `false` 可禁用 Spanner OTEL 指标噪声 |

### 13.4 分步部署

```bash
# 步骤 1：配置 gcloud
gcloud config set project <YOUR_PROJECT_ID>
gcloud auth application-default login

# 步骤 2：一次性基础设施创建
bash deploy/infra/01_setup_gcp.sh

# 步骤 3：上传输入文档（txt 格式）
gcloud storage cp your-docs/*.txt gs://<YOUR_PROJECT>-graphrag-input/documents/

# 步骤 4：构建并推送 Docker 镜像（从仓库根目录执行）
bash deploy/infra/02_build_push.sh

# 步骤 5：创建索引 Job 并运行全量索引
bash deploy/infra/03_deploy_jobs.sh run

# 步骤 6：部署查询服务
bash deploy/infra/04_deploy_query.sh

# 步骤 7：启用 IAP 并授权用户
bash deploy/infra/05_setup_iap.sh
bash deploy/infra/05_setup_iap.sh grant user@your-domain.com

# 增量更新（新增文档后）
bash deploy/infra/03_deploy_jobs.sh update
```

**Docker 构建说明：**
- 两个镜像均从仓库根目录构建，上下文包含完整的 `packages/` 目录。
- 使用 `uv sync --all-packages --no-dev --frozen` 安装所有 workspace 包。
- `query-service` 额外安装 `fastapi` 和 `uvicorn`（不在 workspace 依赖中）。

### 13.5 IAP 访问控制

**授权用户：**

```bash
# 通过部署脚本授权
bash deploy/infra/05_setup_iap.sh grant user@your-domain.com
bash deploy/infra/05_setup_iap.sh grant group:team@your-domain.com

# 或直接使用 gcloud
gcloud iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=graphrag-query-service \
  --region=<REGION> \
  --project=<PROJECT_ID> \
  --member="user:user@your-domain.com" \
  --role="roles/iap.httpsResourceAccessor"
```

> 必须使用 `gcloud iap web add-iam-policy-binding`，**不能**使用 `gcloud run services add-iam-policy-binding`——后者不支持在 Cloud Run 资源上绑定 `roles/iap.httpsResourceAccessor`。

**命令行测试（在 GCP 环境内）：**

```bash
SERVICE_URL="https://<your-cloud-run-url>"

# 从 Metadata Server 获取 ID token
ID_TOKEN=$(curl -s \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${SERVICE_URL}&format=full" \
  -H "Metadata-Flavor: Google")

curl -X POST "$SERVICE_URL/v1/query/global" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "主要讲了什么内容？"}'
```

### 13.6 查询 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 存活探针，服务启动后始终返回 200 |
| `/readyz` | GET | 就绪探针，DataFrames 加载完成后返回 200 + 索引统计；加载中返回 503 |
| `/v1/query/global` | POST | Global search（社区报告级别，无向量查询） |
| `/v1/query/local` | POST | Local search（实体级别 + 向量相似度） |
| `/v1/query/drift` | POST | DRIFT search（渐进式社区精化） |
| `/v1/query/basic` | POST | Basic RAG search（text unit 检索） |

**请求体字段：**

| 字段 | 类型 | 默认值 | 适用范围 |
|------|------|--------|---------|
| `query` | string | 必填 | 全部 |
| `community_level` | int | `2` | global、local、drift |
| `dynamic_community_selection` | bool | `false` | 仅 global |
| `response_type` | string | `"Multiple Paragraphs"` | 全部 |

**示例：**

```bash
# Global search
curl -X POST "$SERVICE_URL/v1/query/global" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "主要角色有哪些？", "community_level": 2}'

# Local search
curl -X POST "$SERVICE_URL/v1/query/local" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "描述一下任务目标。"}'
```

**响应格式：**

```json
{
  "response": "...",
  "context_data": { "entities": [...], "reports": [...] }
}
```

### 13.7 已知限制与注意事项

#### `cache.type: memory`——GCSLiteLLMCache 不可用

`Factory.create()` 在创建 LLM completion 实例时，用 `yaml.dump()` 序列化所有构造参数来生成单例缓存 key。`GCSLiteLLMCache` 内部持有 `google.cloud.storage.Client`，该对象在序列化时抛出 `PicklingError`，导致 `extract_graph` workflow 崩溃。请使用 `cache.type: memory`（进程内缓存，单次运行内有效）。

#### uvloop 与 nest_asyncio2 不兼容

GraphRAG 内部使用 `nest_asyncio2`，它无法 patch uvloop 的事件循环。安装 `uvicorn[standard]`（包含 uvloop）会在启动时报 `ValueError: Can't patch loop of type uvloop.Loop`。请使用不带 `[standard]` 的 `uvicorn`，不安装 uvloop，也不传 `--loop uvloop`。

#### `gemini-3-flash-preview` 仅支持 `VERTEXAI_LOCATION=global`

该模型只在 global endpoint 可用。将 location 设为具体区域（如 `us-central1`）会返回 404。

#### Spanner OTEL 指标噪声

Spanner Python 客户端内置的 OpenTelemetry 指标上报器，在 metric resource labels 不完整（缺少 `instance_id`）时会触发 `400 InvalidArgument` 错误。设置 `SPANNER_ENABLE_BUILT_IN_METRICS=false` 可完全禁用，不影响功能。如保持启用，服务账号需要 `roles/monitoring.metricWriter` 权限。

#### 必须使用单 uvicorn worker

`SpannerResourceManager` 使用模块级单例，不支持多进程共享。必须以 `--workers 1` 运行，通过增加 Cloud Run 实例数量横向扩展。

#### `uv sync` 必须加 `--all-packages`

不加此标志时，workspace 成员包（graphrag、graphrag-storage 等）不会被安装到镜像中。Dockerfile 中正确的命令是 `uv sync --all-packages --no-dev --frozen`。
