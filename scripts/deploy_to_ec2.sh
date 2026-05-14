#!/usr/bin/env bash
# Called by GitHub Actions cd.yml on the EC2 instance after new images are pushed to ECR.
# Performs a rolling update of the production services with zero-downtime.

set -euo pipefail

APP_DIR="/app/market-pulse-predictor"

cd "$APP_DIR"

echo "--- Pulling latest code ---"
git pull origin main

echo "--- Pulling new Docker images ---"
docker compose -f docker-compose.prod.yml pull

echo "--- Restarting services (rolling, no-deps) ---"
docker compose -f docker-compose.prod.yml up -d --remove-orphans

echo "--- Pruning dangling images ---"
docker image prune -f

echo "--- Service status ---"
docker compose -f docker-compose.prod.yml ps
