# Training Guide: GPU Training on Google Colab

## Why Colab?

The RNN/LSTM/GRU models require GPU for reasonable training times. Training on CPU would take hours to days. Google Colab provides free GPU access (T4 GPU, 15GB RAM) which is perfect for this project.

## Architecture Overview

```
Local Machine                  Google Colab              Remote MLflow
    │                              │                          │
    ├─ Data Collection             │                          │
    │  (Airflow DAGs)               │                          │
    │                               │                          │
    ├─ Feature Engineering          │                          │
    │  (data/features/)             │                          │
    │                               │                          │
    └─ Upload features ────────────>│                          │
                                    │                          │
                                    ├─ Train Models            │
                                    │  (GPU accelerated)       │
                                    │                          │
                                    ├─ Log experiments ───────>│
                                    │                          │
                                    └─ Download models ────────>│
                                                                │
    ┌─ Deploy models <──────────────────────────────────────────┘
    │  (API inference)
    │
    └─ Serve predictions
```

## Training Workflow

### Option 1: Remote MLflow with ngrok (Recommended for Development)

**Pros:**
- ✅ Full MLflow tracking maintained
- ✅ Free (no cloud costs)
- ✅ Easy setup
- ✅ Real-time experiment monitoring

**Cons:**
- ⚠️ Requires keeping your laptop running during training
- ⚠️ ngrok free tier has session limits (2 hours)

**Steps:**

1. **Prepare data locally:**
   ```bash
   # Run data collection and feature engineering
   python -m src.ingestion.run_all --lookback-hours 168  # 7 days
   python -m src.sentiment.run_sentiment
   python -m src.features.run_features
   ```

2. **Start MLflow and ngrok:**
   ```bash
   ./scripts/setup_remote_training.sh
   # This will:
   # - Package your features
   # - Start MLflow server
   # - Create ngrok tunnel
   # - Give you the public URL
   ```

3. **Upload to Google Drive:**
   - Upload `colab_upload/features.tar.gz` to your Google Drive
   - Note the ngrok URL from the script output

4. **Open Colab notebook:**
   - Open `notebooks/colab_training.ipynb` in Google Colab
   - Update the `MLFLOW_TRACKING_URI` with your ngrok URL
   - Run all cells

5. **Download trained models:**
   - The notebook will create `trained_models.zip`
   - Download and extract to your local `models/` directory

6. **Verify in MLflow:**
   - Open http://localhost:5001
   - You should see all 9 training runs
   - Models registered in Model Registry

### Option 2: Cloud MLflow (Recommended for Production)

**Pros:**
- ✅ No need to keep laptop running
- ✅ Persistent tracking
- ✅ Production-ready
- ✅ Team collaboration

**Cons:**
- 💰 Small cloud costs (~$5-10/month)

**Setup:**

1. **Deploy MLflow to Railway.app (Free tier available):**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and create project
   railway login
   railway init
   
   # Add PostgreSQL
   railway add postgresql
   
   # Deploy MLflow
   railway up
   ```

2. **Configure environment:**
   ```bash
   # In Colab notebook, use your Railway URL
   os.environ['MLFLOW_TRACKING_URI'] = 'https://your-app.railway.app'
   ```

3. **Follow same training steps as Option 1**

### Option 3: Standalone Colab Training (No MLflow)

**Pros:**
- ✅ Simplest setup
- ✅ No dependencies

**Cons:**
- ❌ No experiment tracking
- ❌ Manual model comparison
- ❌ No model registry

**Only use this if:**
- You're doing a quick test
- You don't need experiment tracking
- You'll manually manage model versions

## Training Time Estimates

On Colab T4 GPU:
- **Per model per task:** ~5-15 minutes
- **All 9 models (3 architectures × 3 tasks):** ~1-2 hours
- **With early stopping:** May finish faster

On CPU (local laptop):
- **Per model:** 2-6 hours ⚠️
- **All 9 models:** 18-54 hours ⚠️

## Model Size and Requirements

- **Each trained model:** ~5-20 MB
- **All 9 models:** ~50-150 MB
- **Feature data:** ~10-100 MB (depends on data collection period)
- **Colab disk space:** 100+ GB available (plenty)
- **Colab RAM:** 12-15 GB (sufficient)

## Troubleshooting

### "Out of Memory" on Colab

Reduce batch size in `params.yaml`:
```yaml
training:
  batch_size: 32  # or even 16
```

### ngrok session expired

Free ngrok sessions last 2 hours. Either:
- Restart ngrok and update Colab notebook
- Use Railway/Render for persistent MLflow

### Models not registering in MLflow

Check that:
1. `MLFLOW_TRACKING_URI` is correct
2. MLflow server is running
3. Network connectivity from Colab to your server

### Feature data not found

Ensure you've run:
```bash
python -m src.features.run_features
```

And the output files exist:
```bash
ls -lh data/features/
# Should show: X_train.npy, y_direction_train.npy, etc.
```

## After Training

### 1. Download Models

Extract `trained_models.zip` to your local `models/` directory:
```bash
unzip trained_models.zip -d .
```

### 2. Verify Models

```bash
ls -lh models/
# Should show: lstm_direction_best.pt, gru_return_best.pt, etc.
```

### 3. Test API Locally

```bash
# Start API
uvicorn src.api.main:app --reload

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### 4. Deploy to Production

Follow Phase 9 deployment guide to deploy to EC2.

## Best Practices

1. **Always version your data with DVC** before training
2. **Tag your MLflow runs** with data version hash
3. **Save training logs** from Colab for debugging
4. **Compare models** in MLflow UI before deploying
5. **Test inference locally** before deploying to production

## Alternative: AWS SageMaker

If you have AWS credits or need production-grade training:

1. Use SageMaker Notebook instances (ml.p3.2xlarge for GPU)
2. MLflow already integrated with SageMaker
3. Direct S3 access for data and models
4. More expensive but more robust

See `docs/sagemaker_training.md` for details (TODO).

## Questions?

- Check MLflow UI for experiment details
- Review Colab notebook output for errors
- Verify feature data shape matches model input
- Ensure all dependencies are installed in Colab
