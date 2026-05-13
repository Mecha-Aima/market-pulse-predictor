#!/bin/bash
set -e

# Railway provides PORT as an environment variable
# Default to 5000 if not set (for local testing)
PORT=${PORT:-5000}

echo "Starting MLflow server on port $PORT"
echo "Database: ${DATABASE_URL:0:20}..."
echo "Artifact root: $ARTIFACT_ROOT"

# Start MLflow server
# NOTE: --serve-artifacts intentionally omitted — it conflicts with gunicorn
# static file routing and causes /static-files/* to 404 (blank UI).
# Clients write artifacts directly to S3 via ARTIFACT_ROOT.
#
# MLflow 2.12.0 compatibility:
#   - --allowed-hosts: added in 3.5.0, NOT available here
#   - --cors-allowed-origins: conflicts with --gunicorn-opts, use env var instead
#   - --workers: replaces --gunicorn-opts for worker count in uvicorn mode

export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*"

exec mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "${ARTIFACT_ROOT:-s3://market-pulse-dvc/mlflow-artifacts}" \
    --workers "${MLFLOW_WORKERS:-2}"
