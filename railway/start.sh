#!/bin/bash
set -e

NGINX_PORT=${PORT:-8080}
MLFLOW_PORT=5001

DATABASE_URL="${DATABASE_URL:?DATABASE_URL must be set}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-s3://market-pulse-dvc/mlflow-artifacts}"

# Resolve static dir (symlinked at build time)
MLFLOW_STATIC_DIR=$( [ -L /mlflow-static ] && echo /mlflow-static || \
    python -c "import mlflow, os; print(os.path.join(os.path.dirname(mlflow.__file__), 'server', 'js', 'build'))" )

# Build nginx config from template
sed -e "s|__PORT__|${NGINX_PORT}|g" \
    -e "s|__MLFLOW_STATIC_DIR__|${MLFLOW_STATIC_DIR}|g" \
    /app/nginx.conf > /tmp/nginx-runtime.conf

echo "Starting nginx on port $NGINX_PORT (proxying MLflow on $MLFLOW_PORT)..."
nginx -c /tmp/nginx-runtime.conf

echo "Starting MLflow on 127.0.0.1:$MLFLOW_PORT..."
export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*"

exec mlflow server \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --gunicorn-opts "--timeout 300 --workers 4"
