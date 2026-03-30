#!/usr/bin/env bash
# Phase 5: Enable IAP on the Cloud Run service and grant user access
#
# PREREQUISITES (one-time manual steps in GCP Console):
#   1. Go to: APIs & Services > OAuth consent screen
#      Configure the OAuth consent screen (User Type: Internal for org users)
#   2. The IAP service account is created automatically by GCP when IAP is first enabled
#
# Usage:
#   bash deploy/infra/05_setup_iap.sh                           # enable IAP only
#   bash deploy/infra/05_setup_iap.sh grant user@example.com    # grant a user access
#   bash deploy/infra/05_setup_iap.sh grant group:team@example.com  # grant a group
set -euo pipefail

PROJECT=grhuang-02
REGION=us-central1
SERVICE_NAME=graphrag-query-service

echo "=== Enabling IAP on Cloud Run service: ${SERVICE_NAME} ==="

# Enable IAP directly on the Cloud Run service (no Load Balancer required)
gcloud run services update "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --iap

echo "  IAP enabled on ${SERVICE_NAME}"

# Grant the Cloud Run IAP service account permission to invoke the service
# (IAP proxies authenticated requests using its own SA)
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT}" --format="value(projectNumber)")
IAP_SA="service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com"

echo "  Granting run.invoker to IAP service account: ${IAP_SA}"
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --member="serviceAccount:${IAP_SA}" \
  --role="roles/run.invoker"

# Handle subcommands
MODE="${1:-}"
MEMBER="${2:-}"

if [[ "${MODE}" == "grant" && -n "${MEMBER}" ]]; then
  # Prefix with "user:" if no prefix given
  if [[ "${MEMBER}" != *":"* ]]; then
    MEMBER="user:${MEMBER}"
  fi
  echo ""
  echo "=== Granting IAP access to: ${MEMBER} ==="
  gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --member="${MEMBER}" \
    --role="roles/iap.httpsResourceAccessor"
  echo "  Access granted"
fi

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" --project="${PROJECT}" \
  --format="value(status.url)")

echo ""
echo "=== IAP setup complete ==="
echo "  Service URL: ${SERVICE_URL}"
echo ""
echo "To grant access to users/groups:"
echo "  bash deploy/infra/05_setup_iap.sh grant user@example.com"
echo "  bash deploy/infra/05_setup_iap.sh grant group:team@your-domain.com"
echo ""
echo "To query (authenticated via IAP):"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  curl -H \"Authorization: Bearer \${TOKEN}\" \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -d '{\"query\": \"What are the main topics?\"}' \\"
echo "       ${SERVICE_URL}/v1/query/global"
