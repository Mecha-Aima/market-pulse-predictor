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
#
# MLflow 2.12.0 + gunicorn multi-worker = broken static file serving.
# /static-files/static/js/*.js → 404. Root cause: WhiteNoise middleware
# doesn't share state across gunicorn workers. Fix: --workers 1.
# --gunicorn-opts passes --timeout 300 for Railway cold-start DB connections.

export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*"

exec mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "${ARTIFACT_ROOT:-s3://market-pulse-dvc/mlflow-artifacts}" \
    --workers 1 \
    --gunicorn-opts "--timeout 300"
