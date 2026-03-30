#!/usr/bin/env bash
# Phase 1: Provision all GCP resources for GraphRAG
# Run once before building/deploying. Safe to re-run (uses --quiet where possible).
#
# Usage: bash deploy/infra/01_setup_gcp.sh
set -euo pipefail

PROJECT=grhuang-02
REGION=us-central1
SPANNER_INSTANCE=graphrag-instance
SPANNER_DB=graphrag-db
REGISTRY=graphrag

QUERY_SA=graphrag-query-sa
INDEXER_SA=graphrag-indexer-sa
QUERY_SA_EMAIL="${QUERY_SA}@${PROJECT}.iam.gserviceaccount.com"
INDEXER_SA_EMAIL="${INDEXER_SA}@${PROJECT}.iam.gserviceaccount.com"

echo "=== [1/6] Enabling required APIs ==="
gcloud services enable \
  run.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iap.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT}"

echo "=== [2/6] Creating GCS buckets ==="
for BUCKET in grhuang-02-graphrag-input grhuang-02-graphrag-cache; do
  if ! gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT}" &>/dev/null; then
    gcloud storage buckets create "gs://${BUCKET}" \
      --project="${PROJECT}" \
      --location="${REGION}" \
      --uniform-bucket-level-access
    echo "  Created gs://${BUCKET}"
  else
    echo "  gs://${BUCKET} already exists, skipping"
  fi
done

# Index bucket gets versioning to protect against accidental overwrites
INDEX_BUCKET=grhuang-02-graphrag-index
if ! gcloud storage buckets describe "gs://${INDEX_BUCKET}" --project="${PROJECT}" &>/dev/null; then
  gcloud storage buckets create "gs://${INDEX_BUCKET}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
  gcloud storage buckets update "gs://${INDEX_BUCKET}" --versioning
  echo "  Created gs://${INDEX_BUCKET} with versioning"
else
  echo "  gs://${INDEX_BUCKET} already exists, skipping"
fi

echo "=== [3/6] Creating Spanner instance and database ==="
if ! gcloud spanner instances describe "${SPANNER_INSTANCE}" --project="${PROJECT}" &>/dev/null; then
  gcloud spanner instances create "${SPANNER_INSTANCE}" \
    --project="${PROJECT}" \
    --config="regional-${REGION}" \
    --edition=ENTERPRISE \
    --processing-units=100 \
    --description="GraphRAG demo"
  echo "  Created Spanner instance ${SPANNER_INSTANCE}"
else
  echo "  Spanner instance ${SPANNER_INSTANCE} already exists, skipping"
fi

if ! gcloud spanner databases describe "${SPANNER_DB}" \
    --instance="${SPANNER_INSTANCE}" --project="${PROJECT}" &>/dev/null; then
  gcloud spanner databases create "${SPANNER_DB}" \
    --instance="${SPANNER_INSTANCE}" \
    --project="${PROJECT}" \
    --database-dialect=GOOGLE_STANDARD_SQL
  echo "  Created Spanner database ${SPANNER_DB}"
else
  echo "  Spanner database ${SPANNER_DB} already exists, skipping"
fi

echo "=== [4/6] Creating Artifact Registry repository ==="
if ! gcloud artifacts repositories describe "${REGISTRY}" \
    --location="${REGION}" --project="${PROJECT}" &>/dev/null; then
  gcloud artifacts repositories create "${REGISTRY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT}" \
    --description="GraphRAG container images"
  echo "  Created Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT}/${REGISTRY}"
else
  echo "  Artifact Registry ${REGISTRY} already exists, skipping"
fi

echo "=== [5/6] Creating Service Accounts ==="
for SA_NAME in "${QUERY_SA}" "${INDEXER_SA}"; do
  SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
      --project="${PROJECT}" \
      --display-name="GraphRAG ${SA_NAME}"
    echo "  Created SA: ${SA_EMAIL}"
  else
    echo "  SA ${SA_EMAIL} already exists, skipping"
  fi
done

echo "=== [6/6] Binding IAM roles ==="

# --- Query SA: read-only on index bucket ---
gcloud storage buckets add-iam-policy-binding "gs://${INDEX_BUCKET}" \
  --member="serviceAccount:${QUERY_SA_EMAIL}" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT}"

# --- Query SA: read+write on cache bucket (read cached responses, write new ones) ---
gcloud storage buckets add-iam-policy-binding "gs://grhuang-02-graphrag-cache" \
  --member="serviceAccount:${QUERY_SA_EMAIL}" \
  --role="roles/storage.objectUser" \
  --project="${PROJECT}"

# --- Query SA: Spanner user (DML only, no DDL — tables are pre-created by indexer) ---
gcloud spanner databases add-iam-policy-binding "${SPANNER_DB}" \
  --instance="${SPANNER_INSTANCE}" \
  --project="${PROJECT}" \
  --member="serviceAccount:${QUERY_SA_EMAIL}" \
  --role="roles/spanner.databaseUser"

# --- Query SA: Vertex AI (call Gemini and text-embedding) ---
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${QUERY_SA_EMAIL}" \
  --role="roles/aiplatform.user"

# --- Indexer SA: read input documents ---
gcloud storage buckets add-iam-policy-binding "gs://grhuang-02-graphrag-input" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT}"

# --- Indexer SA: read+write index bucket (write parquet output) ---
gcloud storage buckets add-iam-policy-binding "gs://${INDEX_BUCKET}" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --project="${PROJECT}"

# --- Indexer SA: read+write cache bucket ---
gcloud storage buckets add-iam-policy-binding "gs://grhuang-02-graphrag-cache" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --project="${PROJECT}"

# --- Indexer SA: Spanner admin (auto-DDL: CREATE TABLE, CREATE VECTOR INDEX) ---
gcloud spanner databases add-iam-policy-binding "${SPANNER_DB}" \
  --instance="${SPANNER_INSTANCE}" \
  --project="${PROJECT}" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/spanner.databaseAdmin"

# --- Indexer SA: Vertex AI (LLM calls during indexing) ---
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/aiplatform.user"

# --- Both SAs: Cloud Monitoring (Spanner client built-in OTEL metrics export) ---
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${QUERY_SA_EMAIL}" \
  --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${INDEXER_SA_EMAIL}" \
  --role="roles/monitoring.metricWriter"

echo ""
echo "=== Infrastructure setup complete ==="
echo "Next step: bash deploy/infra/02_build_push.sh"
