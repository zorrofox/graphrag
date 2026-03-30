#!/usr/bin/env bash
# Phase 4: Deploy the GraphRAG query service to Cloud Run
#
# Usage: bash deploy/infra/04_deploy_query.sh
set -euo pipefail

PROJECT=grhuang-02
REGION=us-central1
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/graphrag"
TAG="${TAG:-latest}"
QUERY_IMAGE="${REGISTRY}/query:${TAG}"
SERVICE_NAME=graphrag-query-service
QUERY_SA="graphrag-query-sa@${PROJECT}.iam.gserviceaccount.com"

ENV_VARS="GRAPHRAG_PROJECT_ID=${PROJECT},\
GCS_BUCKET_INPUT=grhuang-02-graphrag-input,\
GCS_BUCKET_INDEX=grhuang-02-graphrag-index,\
GCS_BUCKET_CACHE=grhuang-02-graphrag-cache,\
SPANNER_INSTANCE_ID=graphrag-instance,\
SPANNER_DATABASE_ID=graphrag-db,\
GOOGLE_CLOUD_PROJECT=${PROJECT},\
VERTEXAI_PROJECT=${PROJECT},\
VERTEXAI_LOCATION=global"

echo "=== Deploying Cloud Run Service: ${SERVICE_NAME} ==="

gcloud run deploy "${SERVICE_NAME}" \
  --image="${QUERY_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --service-account="${QUERY_SA}" \
  --memory=4Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --concurrency=80 \
  --timeout=300 \
  --no-allow-unauthenticated \
  --set-env-vars="${ENV_VARS}"

echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" --project="${PROJECT}" \
  --format="value(status.url)")
echo "=== Service deployed ==="
echo "  URL:  ${SERVICE_URL}"
echo "  Note: Direct access requires IAP (see 05_setup_iap.sh)"
echo ""
echo "  Test liveness (from within GCP network or after IAP setup):"
echo "    curl -H 'Authorization: Bearer \$(gcloud auth print-identity-token)' ${SERVICE_URL}/healthz"
echo ""
echo "Next step: bash deploy/infra/05_setup_iap.sh"
