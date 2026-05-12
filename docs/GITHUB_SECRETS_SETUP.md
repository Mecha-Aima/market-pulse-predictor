# GitHub Secrets Setup Guide

This guide explains how to configure GitHub Secrets for the CI/CD pipeline.

## Required Secrets

The following secrets must be configured in your GitHub repository for the CD workflow to function properly.

### AWS Secrets

#### `AWS_ACCESS_KEY_ID`
- **Description:** AWS Access Key ID for ECR and EC2 access
- **How to get:** 
  1. Go to AWS IAM Console
  2. Create a new IAM user or use existing user
  3. Attach policies: `AmazonEC2ContainerRegistryFullAccess`, `AmazonEC2FullAccess`
  4. Create access key
  5. Copy the Access Key ID

#### `AWS_SECRET_ACCESS_KEY`
- **Description:** AWS Secret Access Key (pair with Access Key ID)
- **How to get:** 
  1. Same process as Access Key ID
  2. Copy the Secret Access Key (only shown once!)
  3. Store securely

#### `AWS_ACCOUNT_ID`
- **Description:** Your AWS Account ID (12-digit number)
- **How to get:**
  1. Go to AWS Console
  2. Click on your account name in top-right
  3. Copy the Account ID (e.g., `123456789012`)

### EC2 Deployment Secrets

#### `EC2_HOST`
- **Description:** Public IP address or hostname of your EC2 instance
- **Example:** `54.123.45.67` or `ec2-54-123-45-67.compute-1.amazonaws.com`
- **How to get:**
  1. Go to EC2 Console
  2. Select your instance
  3. Copy the Public IPv4 address or Public IPv4 DNS

#### `EC2_USER`
- **Description:** SSH username for EC2 instance
- **Default:** `ubuntu` (for Ubuntu instances) or `ec2-user` (for Amazon Linux)
- **Example:** `ubuntu`

#### `EC2_SSH_PRIVATE_KEY`
- **Description:** Private SSH key for EC2 access
- **How to get:**
  1. Use the private key file (.pem) you downloaded when creating the EC2 instance
  2. Copy the entire contents of the file
  3. Include the `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----` lines

**Example format:**
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
(many lines of key data)
...
-----END RSA PRIVATE KEY-----
```

## How to Add Secrets to GitHub

### Via GitHub Web Interface

1. Go to your GitHub repository
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Enter the secret name (exactly as shown above)
6. Paste the secret value
7. Click **Add secret**
8. Repeat for all required secrets

### Via GitHub CLI

```bash
# Install GitHub CLI if not already installed
brew install gh

# Authenticate
gh auth login

# Add secrets
gh secret set AWS_ACCESS_KEY_ID
gh secret set AWS_SECRET_ACCESS_KEY
gh secret set AWS_ACCOUNT_ID
gh secret set EC2_HOST
gh secret set EC2_USER
gh secret set EC2_SSH_PRIVATE_KEY < path/to/private-key.pem
```

## Verification

### Check Secrets are Set

```bash
# List all secrets (values are hidden)
gh secret list
```

Expected output:
```
AWS_ACCESS_KEY_ID        Updated 2024-01-01
AWS_SECRET_ACCESS_KEY    Updated 2024-01-01
AWS_ACCOUNT_ID           Updated 2024-01-01
EC2_HOST                 Updated 2024-01-01
EC2_USER                 Updated 2024-01-01
EC2_SSH_PRIVATE_KEY      Updated 2024-01-01
```

### Test Secret Access in Workflow

The CD workflow includes a test step that verifies secret access. You can trigger it manually:

1. Go to **Actions** tab in GitHub
2. Select **CD - Build and Deploy** workflow
3. Click **Run workflow**
4. Select `main` branch
5. Click **Run workflow**

If secrets are configured correctly, the workflow will proceed. If any secrets are missing, the workflow will fail with an error message indicating which secret is missing.

## Security Best Practices

### 1. Rotate Secrets Regularly
- Rotate AWS access keys every 90 days
- Update GitHub secrets after rotation

### 2. Use Least Privilege
- Create dedicated IAM user for CI/CD
- Only grant necessary permissions
- Don't use root account credentials

### 3. Monitor Secret Usage
- Enable AWS CloudTrail
- Review GitHub Actions logs
- Set up alerts for unusual activity

### 4. Protect SSH Keys
- Never commit private keys to git
- Use strong key encryption
- Rotate EC2 key pairs periodically

### 5. Limit Secret Scope
- Use repository secrets (not organization secrets) when possible
- Only share secrets with workflows that need them

## Troubleshooting

### Error: "AWS credentials not found"
**Solution:** Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set correctly

### Error: "Permission denied (publickey)"
**Solution:** 
- Verify `EC2_SSH_PRIVATE_KEY` contains the complete private key
- Ensure the key matches the EC2 instance's authorized key
- Check `EC2_USER` is correct for your instance type

### Error: "ECR repository does not exist"
**Solution:**
- Create ECR repositories manually:
  ```bash
  aws ecr create-repository --repository-name market-pulse-api
  aws ecr create-repository --repository-name market-pulse-dashboard
  aws ecr create-repository --repository-name market-pulse-training
  ```

### Error: "Host key verification failed"
**Solution:**
- The workflow automatically adds EC2 host to known_hosts
- If issue persists, verify `EC2_HOST` is correct

## ECR Repository Setup

Before running the CD workflow, create the required ECR repositories:

```bash
# Set your AWS region
export AWS_REGION=us-east-1

# Create repositories
aws ecr create-repository \
  --repository-name market-pulse-api \
  --region $AWS_REGION

aws ecr create-repository \
  --repository-name market-pulse-dashboard \
  --region $AWS_REGION

aws ecr create-repository \
  --repository-name market-pulse-training \
  --region $AWS_REGION

# Verify repositories were created
aws ecr describe-repositories --region $AWS_REGION
```

## Testing the Setup

### 1. Test AWS Credentials Locally

```bash
# Configure AWS CLI with your credentials
aws configure

# Test ECR access
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Test EC2 access
ssh -i path/to/private-key.pem ubuntu@<EC2_HOST>
```

### 2. Test GitHub Actions Workflow

1. Make a small change to README.md
2. Commit and push to a feature branch
3. Create a pull request to `main`
4. CI workflow should run automatically
5. After merging to `main`, CD workflow should run

### 3. Verify Deployment

After CD workflow completes:

```bash
# Check API health
curl http://<EC2_HOST>:8000/health

# Check Dashboard
curl http://<EC2_HOST>:8501
```

## Additional Resources

- [GitHub Actions Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [EC2 Key Pairs Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)

## Support

If you encounter issues:
1. Check GitHub Actions logs for detailed error messages
2. Verify all secrets are set correctly
3. Test AWS credentials locally
4. Review the troubleshooting section above
5. Check EC2 instance security groups allow inbound traffic on ports 8000 and 8501

---

**Last Updated:** 2026-05-12  
**Workflow Version:** 1.0
