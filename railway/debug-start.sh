#!/bin/bash
set -e

echo "=== Railway Environment Debug ==="
echo "PORT: ${PORT:-NOT SET}"
echo "DATABASE_URL: ${DATABASE_URL:0:30}..."
echo "ARTIFACT_ROOT: ${ARTIFACT_ROOT:-NOT SET}"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:10}..."
echo "================================="

# Check if PORT is set
if [ -z "$PORT" ]; then
    echo "WARNING: PORT not set by Railway, using default 5000"
    PORT=5000
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL not set!"
    echo "Please add PostgreSQL plugin in Railway"
    exit 1
fi

# Check if ARTIFACT_ROOT is set
if [ -z "$ARTIFACT_ROOT" ]; then
    echo "WARNING: ARTIFACT_ROOT not set, using default"
    ARTIFACT_ROOT="s3://market-pulse-dvc/mlflow-artifacts"
fi

echo ""
echo "Starting MLflow server..."
echo "  Host: 0.0.0.0"
echo "  Port: $PORT"
echo "  Backend: PostgreSQL"
echo "  Artifacts: $ARTIFACT_ROOT"
echo ""

# NOTE: --serve-artifacts is intentionally REMOVED.
#
# Root cause of blank UI: MLflow's --serve-artifacts flag spawns an artifact
# proxy process that conflicts with gunicorn's static file routing, causing
# /static-files/* to return 404. The UI HTML loads fine but all JS/CSS assets
# fail. Clients (Colab) can still write/read artifacts directly via S3 using
# the ARTIFACT_ROOT URI — they don't need the server to proxy artifacts.
#
# IMPORTANT - MLflow 2.12.0 compatibility notes:
#   1. --allowed-hosts was added in MLflow 3.5.0 — DO NOT use it here.
#   2. --cors-allowed-origins conflicts with --gunicorn-opts; use env var instead.
#   3. --workers MUST be 1. MLflow 2.12.0 + gunicorn multi-worker mode breaks
#      static file serving via WhiteNoise — /static-files/static/js/*.js and
#      /static-files/static/css/*.css return 404, causing a blank UI.
#      Single worker avoids this; Railway's free tier doesn't benefit from more.
#   4. --timeout 300 gives the DB connection pool time to init on cold start.

# Set CORS via environment variable (compatible with all server modes in 2.12.0)
export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*"

exec mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --workers 1 \
    --gunicorn-opts "--timeout 300"
