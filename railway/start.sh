#!/bin/bash
set -e

# Railway provides PORT as an environment variable
# Default to 5000 if not set (for local testing)
PORT=${PORT:-5000}

echo "Starting MLflow server on port $PORT"
echo "Database: ${DATABASE_URL:0:20}..."
echo "Artifact root: $ARTIFACT_ROOT"

# Start MLflow server with expanded variables
exec mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "${ARTIFACT_ROOT:-s3://market-pulse-dvc/mlflow-artifacts}" \
    --static-prefix /
