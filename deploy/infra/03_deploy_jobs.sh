#!/usr/bin/env bash
# Phase 3: Create and run the Cloud Run Job for GraphRAG indexing
#
# Usage:
#   bash deploy/infra/03_deploy_jobs.sh          # create job
#   bash deploy/infra/03_deploy_jobs.sh run      # create job AND run immediately
#   bash deploy/infra/03_deploy_jobs.sh update   # trigger incremental update run
set -euo pipefail

PROJECT=grhuang-02
REGION=us-central1
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/graphrag"
TAG="${TAG:-latest}"
INDEXER_IMAGE="${REGISTRY}/indexer:${TAG}"
JOB_NAME=graphrag-indexer
INDEXER_SA="graphrag-indexer-sa@${PROJECT}.iam.gserviceaccount.com"

ENV_VARS="GRAPHRAG_PROJECT_ID=${PROJECT},\
GCS_BUCKET_INPUT=grhuang-02-graphrag-input,\
GCS_BUCKET_INDEX=grhuang-02-graphrag-index,\
GCS_BUCKET_CACHE=grhuang-02-graphrag-cache,\
SPANNER_INSTANCE_ID=graphrag-instance,\
SPANNER_DATABASE_ID=graphrag-db,\
GOOGLE_CLOUD_PROJECT=${PROJECT},\
VERTEXAI_PROJECT=${PROJECT},\
VERTEXAI_LOCATION=global,\
GRAPHRAG_IS_UPDATE=false"

echo "=== Creating / updating Cloud Run Job: ${JOB_NAME} ==="

if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" --project="${PROJECT}" &>/dev/null; then
  echo "  Job exists — updating image"
  gcloud run jobs update "${JOB_NAME}" \
    --image="${INDEXER_IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT}"
else
  gcloud run jobs create "${JOB_NAME}" \
    --image="${INDEXER_IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --service-account="${INDEXER_SA}" \
    --memory=8Gi \
    --cpu=4 \
    --task-timeout=24h \
    --max-retries=1 \
    --parallelism=1 \
    --tasks=1 \
    --set-env-vars="${ENV_VARS}"
  echo "  Created Cloud Run Job: ${JOB_NAME}"
fi

# Handle subcommands
MODE="${1:-}"
case "${MODE}" in
  run)
    echo "=== Triggering full index run (GRAPHRAG_IS_UPDATE=false) ==="
    gcloud run jobs execute "${JOB_NAME}" \
      --region="${REGION}" \
      --project="${PROJECT}" \
      --update-env-vars="GRAPHRAG_IS_UPDATE=false"
    echo "  Job execution started. Monitor at:"
    echo "  https://console.cloud.google.com/run/jobs/${JOB_NAME}/executions?project=${PROJECT}"
    ;;
  update)
    echo "=== Triggering incremental update run (GRAPHRAG_IS_UPDATE=true) ==="
    gcloud run jobs execute "${JOB_NAME}" \
      --region="${REGION}" \
      --project="${PROJECT}" \
      --update-env-vars="GRAPHRAG_IS_UPDATE=true"
    echo "  Update execution started."
    ;;
  *)
    echo ""
    echo "=== Job created. To run: ==="
    echo "  Full index:       bash deploy/infra/03_deploy_jobs.sh run"
    echo "  Incremental:      bash deploy/infra/03_deploy_jobs.sh update"
    echo "  Manual via CLI:   gcloud run jobs execute ${JOB_NAME} --region=${REGION}"
    ;;
esac

echo ""
echo "Next step: bash deploy/infra/04_deploy_query.sh"
