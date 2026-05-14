#!/usr/bin/env bash
# Run this ONCE locally to provision all AWS resources for the project.
# Prerequisites: AWS CLI configured with sufficient IAM permissions.
# Usage: bash scripts/setup_aws_resources.sh

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
APP_NAME="market-pulse"
KEY_NAME="${APP_NAME}-key"
SG_NAME="${APP_NAME}-sg"
INSTANCE_TYPE="t2.micro"
# Ubuntu 22.04 LTS (HVM, SSD) — us-east-1 as of 2024. Verify at:
# https://cloud-images.ubuntu.com/locator/ec2/
AMI_ID="ami-0c7217cdde317cfec"

echo "=== Market Pulse Predictor — AWS Resource Setup ==="
echo "Region: $REGION"
echo ""

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $ACCOUNT_ID"

# ── ECR Repositories ──────────────────────────────────────────────────────────
echo ""
echo "--- Creating ECR repositories ---"
for repo in market-pulse-api market-pulse-dashboard market-pulse-training; do
  if aws ecr describe-repositories --repository-names "$repo" --region "$REGION" &>/dev/null; then
    echo "  ✓ $repo already exists"
  else
    aws ecr create-repository \
      --repository-name "$repo" \
      --region "$REGION" \
      --image-scanning-configuration scanOnPush=true \
      --query 'repository.repositoryUri' --output text
    echo "  ✓ Created $repo"
  fi
done
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ── SSH Key Pair ───────────────────────────────────────────────────────────────
echo ""
echo "--- Creating SSH key pair ---"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
  echo "  ✓ Key pair $KEY_NAME already exists"
else
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query 'KeyMaterial' \
    --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  echo "  ✓ Saved key to $KEY_PATH"
fi

# ── Security Group ─────────────────────────────────────────────────────────────
echo ""
echo "--- Creating security group ---"
if SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null) && \
   [ "$SG_ID" != "None" ]; then
  echo "  ✓ Security group $SG_NAME already exists: $SG_ID"
else
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Market Pulse Predictor security group" \
    --region "$REGION" \
    --query 'GroupId' --output text)

  MY_IP=$(curl -s https://checkip.amazonaws.com)

  # SSH — restricted to current IP
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=SSH}]"
  # HTTP (nginx → api + dashboard)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]"
  # FastAPI direct access
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions "IpProtocol=tcp,FromPort=8000,ToPort=8000,IpRanges=[{CidrIp=0.0.0.0/0}]"
  # Streamlit direct access
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions "IpProtocol=tcp,FromPort=8501,ToPort=8501,IpRanges=[{CidrIp=0.0.0.0/0}]"
  # Airflow UI — restricted to current IP
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions "IpProtocol=tcp,FromPort=8080,ToPort=8080,IpRanges=[{CidrIp=${MY_IP}/32,Description=Airflow}]"

  echo "  ✓ Created security group: $SG_ID"
fi

# ── EC2 Instance ──────────────────────────────────────────────────────────────
echo ""
echo "--- Launching EC2 instance ---"
EXISTING=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${APP_NAME}-prod" "Name=instance-state-name,Values=running,stopped,pending" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

if [ "$EXISTING" != "None" ] && [ -n "$EXISTING" ]; then
  INSTANCE_ID="$EXISTING"
  echo "  ✓ Instance already exists: $INSTANCE_ID"
else
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}-prod}]" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,DeleteOnTermination=true}' \
    --query 'Instances[0].InstanceId' --output text)
  echo "  ✓ Launched instance: $INSTANCE_ID"
  echo "  Waiting for instance to be running..."
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
fi

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  AWS Resources Ready"
echo "======================================================"
echo ""
echo "EC2 Instance ID : $INSTANCE_ID"
echo "EC2 Public IP   : $PUBLIC_IP"
echo "SSH Key         : $KEY_PATH"
echo "ECR Registry    : $ECR_REGISTRY"
echo ""
echo "--- Add these as GitHub Actions Secrets ---"
echo "AWS_ACCOUNT_ID          = $ACCOUNT_ID"
echo "AWS_ACCESS_KEY_ID       = (use your IAM key)"
echo "AWS_SECRET_ACCESS_KEY   = (use your IAM secret)"
echo "EC2_HOST                = $PUBLIC_IP"
echo "EC2_USER                = ubuntu"
echo "EC2_SSH_PRIVATE_KEY     = \$(cat $KEY_PATH)"
echo ""
echo "--- Next step ---"
echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'bash -s' < scripts/setup_ec2.sh"
