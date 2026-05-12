# Railway Deployment Quick Fix

## Error: "No start command detected"

You're seeing this because Railway is trying to auto-detect your app instead of using the Dockerfile.

## Quick Fix (Choose One)

### Fix 1: Force Dockerfile Build (Recommended)

1. **In Railway Dashboard:**
   - Go to your service
   - Click **Settings** (left sidebar)
   - Click **Build** section
   - Change **Builder** to: `Dockerfile`
   - Set **Dockerfile Path** to: `railway/Dockerfile.mlflow`
   - Click **Save**

2. **Redeploy:**
   - Go to **Deployments** tab
   - Click **Redeploy** on the latest deployment

### Fix 2: Set Start Command Manually

1. **In Railway Dashboard:**
   - Go to your service
   - Click **Settings**
   - Click **Deploy** section
   - Set **Start Command** to:
     ```
     mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri $DATABASE_URL --default-artifact-root $ARTIFACT_ROOT
     ```
   - Click **Save**

2. **Add Install Command:**
   - In same **Deploy** section
   - Set **Install Command** to:
     ```
     pip install mlflow==2.12.0 psycopg2-binary boto3 pymysql
     ```
   - Click **Save**

3. **Redeploy**

### Fix 3: Use Railway CLI

```bash
# In your project directory
railway up --dockerfile railway/Dockerfile.mlflow
```

## Verify Deployment

After redeploying, check:

1. **Build Logs:**
   - Should show "Building with Dockerfile" or similar
   - Should install MLflow dependencies
   - Should complete successfully

2. **Deploy Logs:**
   - Should show MLflow server starting
   - Should show: "Listening at: http://0.0.0.0:$PORT"

3. **Test Health:**
   ```bash
   curl https://your-app.up.railway.app/health
   ```
   Should return: `{"status": "ok"}`

## Still Not Working?

### Check Environment Variables

Ensure these are set in Railway:

```bash
DATABASE_URL=postgresql://...  # Auto-set by Railway PostgreSQL
ARTIFACT_ROOT=s3://market-pulse-dvc/mlflow-artifacts
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=us-east-1
```

### Check PostgreSQL

1. Verify PostgreSQL plugin is added
2. Check that `DATABASE_URL` is set automatically
3. Format should be: `postgresql://user:pass@host:port/db`

### View Logs

```bash
# Using Railway CLI
railway logs

# Or in dashboard: Deployments → Click deployment → View logs
```

## Common Issues

### "Port already in use"
- Railway sets `$PORT` automatically
- Make sure your start command uses `$PORT` not hardcoded `5000`

### "Database connection failed"
- Check PostgreSQL is running
- Verify `DATABASE_URL` format
- Try connecting manually: `railway run psql $DATABASE_URL`

### "S3 access denied"
- Verify AWS credentials are correct
- Check IAM permissions include S3 read/write
- Test: `aws s3 ls s3://market-pulse-dvc/`

## Success Checklist

- [ ] Build completes without errors
- [ ] Deploy shows MLflow starting
- [ ] Health endpoint returns 200
- [ ] Can access MLflow UI at your Railway URL
- [ ] Can create experiments via API
- [ ] Models can be logged from Colab

## Next Steps

Once deployed successfully:

1. Copy your Railway URL
2. Update Colab notebook with the URL
3. Test connection from Colab
4. Start training!

## Need More Help?

- Full guide: [RAILWAY_SETUP.md](RAILWAY_SETUP.md)
- Railway docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
