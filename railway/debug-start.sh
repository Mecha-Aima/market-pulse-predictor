#!/bin/bash
set -e

echo "=== Railway Environment Debug ==="
echo "PORT: ${PORT:-NOT SET}"
echo "DATABASE_URL: ${DATABASE_URL:0:30}..."
echo "ARTIFACT_ROOT: ${ARTIFACT_ROOT:-NOT SET}"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:10}..."
echo "================================="

# ── Port setup ──────────────────────────────────────────────────────────────
# Railway sets PORT. Nginx listens on $PORT (public), MLflow on 5001 (internal).
NGINX_PORT=${PORT:-8080}
MLFLOW_PORT=5001

if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL not set! Add the PostgreSQL plugin in Railway."
    exit 1
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-s3://market-pulse-dvc/mlflow-artifacts}"

# ── Find the MLflow static asset directory ──────────────────────────────────
# This was symlinked to /mlflow-static during the Docker build, but we
# resolve it at runtime too for safety.
if [ -L "/mlflow-static" ]; then
    MLFLOW_STATIC_DIR="/mlflow-static"
else
    MLFLOW_STATIC_DIR=$(python -c "import mlflow, os; print(os.path.join(os.path.dirname(mlflow.__file__), 'server', 'js', 'build'))")
fi
echo "MLflow static dir: $MLFLOW_STATIC_DIR"

# ── Configure nginx ─────────────────────────────────────────────────────────
# Substitute runtime values into the nginx template.
# PORT is injected by Railway and can change between deploys.
sed \
    -e "s|__PORT__|${NGINX_PORT}|g" \
    -e "s|__MLFLOW_STATIC_DIR__|${MLFLOW_STATIC_DIR}|g" \
    /app/nginx.conf > /tmp/nginx-runtime.conf

echo ""
echo "Starting nginx..."
echo "  Public port:   $NGINX_PORT"
echo "  MLflow port:   $MLFLOW_PORT (internal)"
echo "  Static dir:    $MLFLOW_STATIC_DIR"
echo ""

nginx -c /tmp/nginx-runtime.conf
echo "nginx started."

# ── Start MLflow ─────────────────────────────────────────────────────────────
# MLflow binds to 127.0.0.1:5001 (internal only — nginx proxies to it).
# --workers 4 is the MLflow default and is fine for single-container use.
# Static files at /static-files/static/* are served by nginx, not Flask,
# so the Flask /static/ route conflict is completely bypassed.
echo ""
echo "Starting MLflow server..."
echo "  Backend:   PostgreSQL"
echo "  Artifacts: $ARTIFACT_ROOT"
echo ""

export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*"

exec mlflow server \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --gunicorn-opts "--timeout 300 --workers 4"
