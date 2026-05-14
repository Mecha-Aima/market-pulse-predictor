# Production Deployment Guide

## Architecture

```
GitHub (push to main)
  └─► GitHub Actions cd.yml
        ├─ Build + push to ECR (api, dashboard, training)
        └─ SSH to EC2 → docker-compose.prod.yml up

EC2 t2.micro — Ubuntu 22.04
  ├─ nginx       :80    → /api/* → api:8000, /* → dashboard:8501
  ├─ api         :8000  (FastAPI, models loaded from ./models/ volume)
  ├─ dashboard   :8501  (Streamlit)
  └─ Airflow stack (separate compose)
        ├─ airflow-webserver  :8080
        ├─ airflow-scheduler
        └─ postgres           :5432

External services
  ├─ Railway  → MLflow tracking server
  └─ S3       → DVC remote (data + model artifacts)
```

---

## Scripts Reference

| Script | When to run | Called by |
|--------|-------------|-----------|
| `scripts/setup_aws_resources.sh` | Once, locally, to provision AWS infra | You, manually |
| `scripts/setup_ec2.sh` | Once, on fresh EC2 via SSH | You, manually |
| `scripts/deploy_to_ec2.sh` | Every push to `main` | `cd.yml` via SSH |

---

## One-Time Setup

### Step 1 — Provision AWS Resources

Run locally with the correct AWS profile.

```bash
export AWS_PROFILE=your-profile-name
export AWS_DEFAULT_REGION=us-east-1

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Using account: $ACCOUNT_ID"

# ECR repositories
aws ecr create-repository --repository-name market-pulse-api \
  --region us-east-1 --image-scanning-configuration scanOnPush=true

aws ecr create-repository --repository-name market-pulse-dashboard \
  --region us-east-1 --image-scanning-configuration scanOnPush=true

aws ecr create-repository --repository-name market-pulse-training \
  --region us-east-1 --image-scanning-configuration scanOnPush=true

# SSH key pair
aws ec2 create-key-pair --key-name market-pulse-key --region us-east-1 \
  --query 'KeyMaterial' --output text > ~/.ssh/market-pulse-key.pem
chmod 400 ~/.ssh/market-pulse-key.pem

# Security group
MY_IP=$(curl -s https://checkip.amazonaws.com)

SG_ID=$(aws ec2 create-security-group \
  --group-name market-pulse-sg \
  --description "Market Pulse Predictor" \
  --region us-east-1 \
  --query 'GroupId' --output text)
echo "Security Group: $SG_ID"

# SSH — your IP only
aws ec2 authorize-security-group-ingress --group-id $SG_ID --region us-east-1 \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32}]"

# HTTP (nginx → api + dashboard) — public
aws ec2 authorize-security-group-ingress --group-id $SG_ID --region us-east-1 \
  --ip-permissions "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]"

# FastAPI direct access — public
aws ec2 authorize-security-group-ingress --group-id $SG_ID --region us-east-1 \
  --ip-permissions "IpProtocol=tcp,FromPort=8000,ToPort=8000,IpRanges=[{CidrIp=0.0.0.0/0}]"

# Streamlit direct access — public
aws ec2 authorize-security-group-ingress --group-id $SG_ID --region us-east-1 \
  --ip-permissions "IpProtocol=tcp,FromPort=8501,ToPort=8501,IpRanges=[{CidrIp=0.0.0.0/0}]"

# Airflow UI — your IP only
aws ec2 authorize-security-group-ingress --group-id $SG_ID --region us-east-1 \
  --ip-permissions "IpProtocol=tcp,FromPort=8080,ToPort=8080,IpRanges=[{CidrIp=${MY_IP}/32}]"

# Launch t2.micro (Ubuntu 22.04 LTS — us-east-1)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t2.micro \
  --key-name market-pulse-key \
  --security-group-ids $SG_ID \
  --region us-east-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=market-pulse-prod}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,DeleteOnTermination=true}' \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID --region us-east-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "========================================"
echo "  ECR Registry : ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"
echo "  Instance ID  : $INSTANCE_ID"
echo "  Public IP    : $PUBLIC_IP"
echo "  SSH key      : ~/.ssh/market-pulse-key.pem"
echo "========================================"
```

---

### Step 2 — Push Models to S3

So the EC2 instance can `dvc pull` them at startup.

```bash
cd /path/to/market-pulse-predictor
dvc push
```

---

### Step 3 — Bootstrap the EC2 Instance

Wait ~60 seconds after launch for SSH to become available.

```bash
ssh -i ~/.ssh/market-pulse-key.pem -o StrictHostKeyChecking=no ubuntu@<PUBLIC_IP> \
  'bash -s' < scripts/setup_ec2.sh
```

This installs Docker, 2 GB swap, AWS CLI, DVC, clones the repo, and creates a `.env` placeholder.

---

### Step 4 — Set GitHub Actions Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|-------------|-------|
| `AWS_ACCOUNT_ID` | Your 12-digit account ID from Step 1 |
| `AWS_ACCESS_KEY_ID` | IAM access key with ECR + EC2 permissions |
| `AWS_SECRET_ACCESS_KEY` | Matching IAM secret key |
| `EC2_HOST` | `PUBLIC_IP` from Step 1 |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_PRIVATE_KEY` | Full contents of `cat ~/.ssh/market-pulse-key.pem` |

---

### Step 5 — Configure and Start Services on EC2

```bash
# Copy your production .env to the instance
scp -i ~/.ssh/market-pulse-key.pem .env ubuntu@<PUBLIC_IP>:/app/market-pulse-predictor/.env

# SSH in
ssh -i ~/.ssh/market-pulse-key.pem ubuntu@<PUBLIC_IP>
```

Once on the instance:

```bash
cd /app/market-pulse-predictor

# Pull models + data from S3
dvc pull

# Apply docker group membership without logout
newgrp docker

# Start production services (nginx + api + dashboard)
docker compose -f docker-compose.prod.yml up -d

# One-time Airflow DB init
docker compose -f docker-compose.airflow.yml run --rm airflow-init

# Start Airflow scheduler (always on) + webserver (access at :8080)
docker compose -f docker-compose.airflow.yml up -d airflow-scheduler airflow-webserver
```

---

### Step 6 — Trigger the CD Pipeline

Push any commit to `main`. The CD workflow will:
1. Build all three Docker images
2. Push them to ECR
3. SSH to EC2 and run `scripts/deploy_to_ec2.sh`
4. Health-check `GET /api/v1/health`

---

## Verifying the Deployment

```bash
# API health
curl http://<PUBLIC_IP>/api/v1/health
# Expected: {"status":"ok","model_loaded":true,...}

# Prediction
curl -X POST http://<PUBLIC_IP>/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Streamlit dashboard
open http://<PUBLIC_IP>

# Airflow UI (your IP must be in the security group)
open http://<PUBLIC_IP>:8080
# Default credentials: admin / admin
```

---

## Updating the .env on EC2

```bash
scp -i ~/.ssh/market-pulse-key.pem .env ubuntu@<PUBLIC_IP>:/app/market-pulse-predictor/.env
ssh -i ~/.ssh/market-pulse-key.pem ubuntu@<PUBLIC_IP> \
  "cd /app/market-pulse-predictor && docker compose -f docker-compose.prod.yml up -d"
```

---

## Checking Logs

```bash
ssh -i ~/.ssh/market-pulse-key.pem ubuntu@<PUBLIC_IP>

# Service logs
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f dashboard
docker compose -f docker-compose.prod.yml logs -f nginx

# Airflow logs
docker compose -f docker-compose.airflow.yml logs -f airflow-scheduler

# All running containers
docker ps
```

---

## Airflow DAG Schedule

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `ingestion_dag` | Mon–Fri 21:00 UTC | Fetch price + sentiment data, DVC snapshot |
| `sentiment_dag` | Hourly at :05 | Run VADER on new raw records |
| `feature_dag` | Hourly at :15 | Build time-series feature arrays |
| `training_dag` | Daily at 02:00 UTC | Retrain all 9 models, register best in MLflow |

---

## Re-enabling Your SSH Access After IP Change

```bash
export AWS_PROFILE=your-profile-name
MY_NEW_IP=$(curl -s https://checkip.amazonaws.com)
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=market-pulse-sg" \
  --query 'SecurityGroups[0].GroupId' --output text)

# Remove old SSH rule and add new one
aws ec2 revoke-security-group-ingress --group-id $SG_ID \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=0.0.0.0/0}]" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_NEW_IP}/32}]"
```

---

## Cost Estimate (t2.micro free tier)

| Service | Cost |
|---------|------|
| EC2 t2.micro | Free (750 hrs/month, first 12 months) |
| EBS 20 GB | Free (30 GB/month included) |
| ECR storage | Free (500 MB/month included) |
| S3 (DVC remote) | ~$0.02/GB/month |
| Railway (MLflow) | Free tier / $5/month hobby |
| **Total** | **~$0–5/month** |
