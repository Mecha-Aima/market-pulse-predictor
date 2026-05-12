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

# Start MLflow server
exec mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "$ARTIFACT_ROOT"
