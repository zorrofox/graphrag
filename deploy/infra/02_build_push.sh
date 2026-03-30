#!/usr/bin/env bash
# Phase 2: Build Docker images and push to Artifact Registry
#
# Run from the repository root:
#   bash deploy/infra/02_build_push.sh
#
# Optional: pass a custom tag (default: latest)
#   TAG=v1.0.0 bash deploy/infra/02_build_push.sh
set -euo pipefail

PROJECT=grhuang-02
REGION=us-central1
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/graphrag"
TAG="${TAG:-latest}"

QUERY_IMAGE="${REGISTRY}/query:${TAG}"
INDEXER_IMAGE="${REGISTRY}/indexer:${TAG}"

echo "=== Configuring Docker auth for Artifact Registry ==="
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "=== Building query service image: ${QUERY_IMAGE} ==="
docker build \
  --file deploy/query-service/Dockerfile \
  --tag "${QUERY_IMAGE}" \
  --platform linux/amd64 \
  .

echo "=== Building indexer image: ${INDEXER_IMAGE} ==="
docker build \
  --file deploy/indexer/Dockerfile \
  --tag "${INDEXER_IMAGE}" \
  --platform linux/amd64 \
  .

echo "=== Pushing images to Artifact Registry ==="
docker push "${QUERY_IMAGE}"
docker push "${INDEXER_IMAGE}"

echo ""
echo "=== Images pushed ==="
echo "  Query:   ${QUERY_IMAGE}"
echo "  Indexer: ${INDEXER_IMAGE}"
echo ""
echo "Next step: bash deploy/infra/03_deploy_jobs.sh"
