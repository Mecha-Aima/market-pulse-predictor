# Railway.app MLflow Setup Guide

## Prerequisites

1. Railway.app account (free tier available)
2. AWS S3 bucket for artifact storage (from your existing setup)
3. Railway CLI (optional but recommended)

## Setup Steps

### Important: Railway Configuration Files

This project includes two configuration options for Railway:

1. **railway.toml** (in project root) - Tells Railway to use the Dockerfile
2. **nixpacks.toml** (in project root) - Alternative using Nixpacks (Railway's default)

Both are configured. Railway will automatically detect and use them.

### Option A: Using Railway Web UI (Easiest)

#### 1. Create New Project

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Connect your GitHub account
4. Select `market-pulse-predictor` repository
5. Railway will detect the Dockerfile

#### 2. Add PostgreSQL Database

1. In your Railway project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically provision a PostgreSQL instance
4. The `DATABASE_URL` environment variable is automatically set

#### 3. Configure Environment Variables

In your Railway project settings, add these variables:

```bash
# Artifact Storage (use your existing S3 bucket)
ARTIFACT_ROOT=s3://market-pulse-dvc/mlflow-artifacts
AWS_ACCESS_KEY_ID=<your-aws-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret>
AWS_DEFAULT_REGION=us-east-1

# Database (automatically set by Railway, but verify)
DATABASE_URL=postgresql://user:pass@host:port/db

# Optional: Basic Auth for MLflow UI
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=<generate-strong-password>
```

#### 4. Configure Build Settings

Railway should automatically detect the `railway.toml` or `nixpacks.toml` file.

**If using Dockerfile (recommended):**
1. Go to "Settings" → "Build"
2. Verify **Builder**: "Dockerfile"
3. Verify **Dockerfile Path**: `railway/Dockerfile.mlflow`
4. Click "Save"

**If Railway doesn't detect the config:**
1. Go to "Settings" → "Build"  
2. Manually set **Builder**: "Dockerfile"
3. Set **Dockerfile Path**: `railway/Dockerfile.mlflow`
4. Click "Save"

**Alternative - Using Nixpacks:**
If you prefer Nixpacks over Docker:
1. The `nixpacks.toml` in the root will be auto-detected
2. No manual configuration needed

#### 5. Deploy

1. Click "Deploy" or push to your GitHub repo
2. Railway will build and deploy automatically
3. Wait for deployment to complete (~2-3 minutes)
4. Copy your public URL (e.g., `https://your-app.up.railway.app`)

#### 6. Verify Deployment

```bash
# Test health endpoint
curl https://your-app.up.railway.app/health

# Should return: {"status": "ok"}
```

### Option B: Using Railway CLI (Advanced)

#### 1. Install Railway CLI

```bash
# macOS
brew install railway

# Or using npm
npm install -g @railway/cli
```

#### 2. Login and Initialize

```bash
# Login to Railway
railway login

# Navigate to your project
cd /path/to/market-pulse-predictor

# Link to Railway project (or create new)
railway link

# Or create new project
railway init
```

#### 3. Add PostgreSQL

```bash
railway add --database postgresql
```

#### 4. Set Environment Variables

```bash
# Set artifact storage
railway variables set ARTIFACT_ROOT=s3://market-pulse-dvc/mlflow-artifacts
railway variables set AWS_ACCESS_KEY_ID=<your-key>
railway variables set AWS_SECRET_ACCESS_KEY=<your-secret>
railway variables set AWS_DEFAULT_REGION=us-east-1

# Optional: Basic auth
railway variables set MLFLOW_TRACKING_USERNAME=admin
railway variables set MLFLOW_TRACKING_PASSWORD=<password>
```

#### 5. Deploy

```bash
# Deploy from current directory
railway up

# Or deploy specific Dockerfile
railway up --dockerfile railway/Dockerfile.mlflow
```

#### 6. Get Public URL

```bash
railway domain
# Copy the URL shown
```

## Post-Deployment Configuration

### 1. Update Local .env

Add your Railway MLflow URL to `.env`:

```bash
MLFLOW_TRACKING_URI=https://your-app.up.railway.app
```

### 2. Test Connection Locally

```bash
python -c "
import mlflow
import os

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
client = mlflow.tracking.MlflowClient()
print('✅ Connected to MLflow!')
print(f'Experiments: {len(client.search_experiments())}')
"
```

### 3. Create Experiment

```bash
python -c "
import mlflow
import os

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment('market_pulse')
print('✅ Experiment created!')
"
```

## Cost Estimates

### Railway Free Tier
- ✅ $5 free credit per month
- ✅ Enough for development/testing
- ✅ Automatic sleep after inactivity (wakes on request)

### Railway Hobby Plan ($5/month)
- ✅ No sleep
- ✅ 500 hours execution time
- ✅ Recommended for production

### AWS S3 Costs
- ~$0.023 per GB storage
- ~$0.09 per GB transfer
- Estimated: $1-5/month for this project

**Total estimated cost: $0-10/month**

## Troubleshooting

### Issue: "No start command detected" Error

**Error message:**
```
✖ No start command detected. Specify a start command
railpack process exited with an error
```

**Cause:** Railway is using Railpack instead of your Dockerfile.

**Solution:**

**Option 1: Use railway.toml (Recommended)**
1. Ensure `railway.toml` exists in your project root (it should)
2. In Railway dashboard: Settings → Build → Redeploy
3. Railway should now detect the Dockerfile

**Option 2: Manual Configuration**
1. Go to Railway dashboard → Your service → Settings
2. Click "Build" section
3. Change **Builder** from "Nixpacks" to "Dockerfile"
4. Set **Dockerfile Path**: `railway/Dockerfile.mlflow`
5. Click "Save"
6. Go to "Deployments" → Click "Redeploy"

**Option 3: Use Nixpacks**
1. Ensure `nixpacks.toml` exists in root (it should)
2. In Railway dashboard: Settings → Build
3. Keep **Builder** as "Nixpacks"
4. Redeploy

**Option 4: Set Start Command Manually**
1. Go to Settings → Deploy
2. Set **Start Command**: 
   ```
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri $DATABASE_URL --default-artifact-root $ARTIFACT_ROOT
   ```
3. Save and redeploy

### Issue: "Connection refused"

**Solution:** Check that:
1. Railway service is running (not sleeping)
2. Health check is passing
3. Port 5000 is exposed
4. DATABASE_URL is set correctly

```bash
# Check logs
railway logs
```

### Issue: "S3 access denied"

**Solution:** Verify AWS credentials:
1. Check IAM permissions include S3 read/write
2. Verify bucket name is correct
3. Check region matches

```bash
# Test S3 access
aws s3 ls s3://market-pulse-dvc/mlflow-artifacts/
```

### Issue: "Database connection failed"

**Solution:**
1. Verify PostgreSQL is running in Railway
2. Check DATABASE_URL format: `postgresql://user:pass@host:port/db`
3. Ensure Railway PostgreSQL plugin is added

### Issue: "Slow response times"

**Solution:**
1. Upgrade to Hobby plan (no sleep)
2. Or accept 30-second cold start on free tier
3. Keep service warm with periodic pings

## Security Best Practices

### 1. Enable Basic Auth (Recommended)

Add to environment variables:
```bash
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=<strong-password>
```

Update Dockerfile:
```dockerfile
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${DATABASE_URL} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --app-name basic-auth
```

### 2. Use Railway Private Networking (Hobby+ plan)

For production, use Railway's private networking to restrict access.

### 3. Rotate Credentials Regularly

- Change MLFLOW_TRACKING_PASSWORD monthly
- Rotate AWS keys quarterly
- Use AWS IAM roles when possible

## Monitoring

### View Logs

```bash
# Real-time logs
railway logs --follow

# Or in web UI: Project → Deployments → Logs
```

### Check Metrics

Railway dashboard shows:
- CPU usage
- Memory usage
- Network traffic
- Request count

### Set Up Alerts

1. Go to Project Settings → Notifications
2. Add webhook or email for deployment failures
3. Monitor health check failures

## Backup and Recovery

### Database Backup

Railway automatically backs up PostgreSQL daily. To manually backup:

```bash
# Export experiments
railway run python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
print(f'Experiments: {len(experiments)}')
"
```

### Artifact Backup

Your artifacts are in S3, which has built-in durability. Enable S3 versioning:

```bash
aws s3api put-bucket-versioning \
    --bucket market-pulse-dvc \
    --versioning-configuration Status=Enabled
```

## Next Steps

1. ✅ Deploy MLflow to Railway
2. ✅ Test connection from local machine
3. ✅ Update Colab notebook with Railway URL
4. ✅ Run training on Colab
5. ✅ Verify experiments appear in Railway MLflow UI
6. ✅ Update production .env with Railway URL

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- MLflow Docs: https://mlflow.org/docs/latest/index.html
