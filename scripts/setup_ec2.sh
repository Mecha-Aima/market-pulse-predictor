#!/usr/bin/env bash
# Bootstrap script — run once on a fresh EC2 Ubuntu 22.04 instance.
# Usage (from local machine):
#   ssh -i ~/.ssh/market-pulse-key.pem ubuntu@<EC2_IP> 'bash -s' < scripts/setup_ec2.sh

set -euo pipefail

REPO_URL="https://github.com/mecha-aima/market-pulse-predictor.git"
APP_DIR="/app/market-pulse-predictor"

echo "=== EC2 Bootstrap: Market Pulse Predictor ==="

# ── System packages ────────────────────────────────────────────────────────────
echo ""
echo "--- Installing system packages ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates gnupg lsb-release unzip python3-pip

# ── Docker CE ─────────────────────────────────────────────────────────────────
echo ""
echo "--- Installing Docker CE ---"
if ! command -v docker &>/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker ubuntu
  echo "  ✓ Docker installed"
else
  echo "  ✓ Docker already installed"
fi

# ── AWS CLI ───────────────────────────────────────────────────────────────────
echo ""
echo "--- Installing AWS CLI ---"
if ! command -v aws &>/dev/null; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp/
  sudo /tmp/aws/install
  rm -rf /tmp/awscliv2.zip /tmp/aws
  echo "  ✓ AWS CLI installed"
else
  echo "  ✓ AWS CLI already installed"
fi

# ── DVC ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- Installing DVC ---"
pip3 install --quiet "dvc[s3]>=3.49"
echo "  ✓ DVC installed"

# ── Swap (2 GB) — critical for t2.micro with PyTorch ──────────────────────────
echo ""
echo "--- Configuring 2 GB swap ---"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "  ✓ Swap configured (2 GB)"
else
  echo "  ✓ Swap already configured"
fi

# ── Clone repository ──────────────────────────────────────────────────────────
echo ""
echo "--- Cloning repository ---"
sudo mkdir -p /app
sudo chown ubuntu:ubuntu /app
if [ -d "$APP_DIR" ]; then
  echo "  ✓ Repo already cloned, pulling latest..."
  cd "$APP_DIR" && git pull origin main
else
  git clone "$REPO_URL" "$APP_DIR"
  echo "  ✓ Cloned to $APP_DIR"
fi

# ── Git identity for DVC auto-commits ─────────────────────────────────────────
cd "$APP_DIR"
git config user.name "Airflow Bot"
git config user.email "airflow@market-pulse.local"

# ── .env reminder ─────────────────────────────────────────────────────────────
echo ""
echo "--- Checking .env ---"
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "  ⚠  .env created from .env.example — populate production values:"
  echo "      nano $APP_DIR/.env"
else
  echo "  ✓ .env exists"
fi

echo ""
echo "======================================================"
echo "  EC2 Bootstrap Complete"
echo "======================================================"
echo ""
echo "Next steps:"
echo "  1. Edit $APP_DIR/.env with production values"
echo "     (MLFLOW_TRACKING_URI, AWS creds, REDDIT creds)"
echo ""
echo "  2. Pull data and models from S3:"
echo "     cd $APP_DIR && dvc pull"
echo ""
echo "  3. Start production services:"
echo "     cd $APP_DIR"
echo "     newgrp docker"
echo "     docker compose -f docker-compose.prod.yml up -d"
echo ""
echo "  4. Init Airflow (one-time):"
echo "     docker compose -f docker-compose.airflow.yml run --rm airflow-init"
echo "     docker compose -f docker-compose.airflow.yml up -d airflow-scheduler airflow-webserver"
