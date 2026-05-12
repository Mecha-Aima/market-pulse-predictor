# Training Architecture: Visual Guide

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR LOCAL MACHINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Airflow    │───▶│  Sentiment   │───▶│   Feature    │        │
│  │  Ingestion   │    │   Analysis   │    │  Engineering │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                    │                    │                │
│         ▼                    ▼                    ▼                │
│  ┌─────────────────────────────────────────────────────┐          │
│  │              data/features/                          │          │
│  │  • X_train.npy, X_val.npy, X_test.npy              │          │
│  │  • y_direction_*.npy, y_return_*.npy, ...          │          │
│  │  • feature_columns.json, scaler.pkl                │          │
│  └─────────────────────────────────────────────────────┘          │
│                           │                                         │
│                           │ Package & Upload                        │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────┐          │
│  │         colab_upload/features.tar.gz                │          │
│  └─────────────────────────────────────────────────────┘          │
│                           │                                         │
│  ┌─────────────────────────────────────────────────────┐          │
│  │         MLflow Server (localhost:5001)              │          │
│  │  • Experiment tracking                              │          │
│  │  • Model registry                                   │          │
│  │  • Artifact storage                                 │          │
│  └─────────────────────────────────────────────────────┘          │
│                           │                                         │
│                           │ Expose via ngrok                        │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────┐          │
│  │      https://abc123.ngrok.io                        │          │
│  │      (Public MLflow endpoint)                       │          │
│  └─────────────────────────────────────────────────────┘          │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            │ Internet
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                           │         GOOGLE COLAB                    │
├───────────────────────────┼─────────────────────────────────────────┤
│                           │                                         │
│  ┌────────────────────────▼──────────────────────────┐            │
│  │         Download features.tar.gz                  │            │
│  │         from Google Drive                         │            │
│  └────────────────────────┬──────────────────────────┘            │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────┐          │
│  │         Extract to data/features/                   │          │
│  └─────────────────────────────────────────────────────┘          │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────┐          │
│  │         Training Loop (GPU Accelerated)             │          │
│  │                                                     │          │
│  │  For each model in [RNN, LSTM, GRU]:               │          │
│  │    For each task in [direction, return, vol]:      │          │
│  │      1. Load data                                   │          │
│  │      2. Initialize model                            │          │
│  │      3. Train with early stopping                   │          │
│  │      4. Evaluate on test set                        │          │
│  │      5. Log to MLflow ──────────────────────┐      │          │
│  │      6. Save checkpoint                      │      │          │
│  │                                              │      │          │
│  │  Total: 9 models trained                     │      │          │
│  │  Time: ~1-2 hours on T4 GPU                  │      │          │
│  └──────────────────────────────────────────────┼──────┘          │
│                           │                     │                  │
│                           │                     │                  │
│                           ▼                     │                  │
│  ┌─────────────────────────────────────────────┼──────┐          │
│  │         models/                              │      │          │
│  │  • rnn_direction_best.pt                     │      │          │
│  │  • lstm_direction_best.pt                    │      │          │
│  │  • gru_direction_best.pt                     │      │          │
│  │  • ... (9 total)                             │      │          │
│  └─────────────────────────────────────────────┼──────┘          │
│                           │                     │                  │
│                           │ Download            │                  │
│                           ▼                     │                  │
│  ┌─────────────────────────────────────────────┼──────┐          │
│  │         trained_models.zip                   │      │          │
│  └─────────────────────────────────────────────┼──────┘          │
│                                                 │                  │
└─────────────────────────────────────────────────┼──────────────────┘
                                                  │
                                                  │ MLflow Logs
                                                  │
┌─────────────────────────────────────────────────▼──────────────────┐
│                    MLFLOW SERVER (via ngrok)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              Experiment: market_pulse                │        │
│  │                                                      │        │
│  │  Run 1: rnn_direction_20240515_1400                 │        │
│  │    • Params: hidden_size=128, lr=0.001, ...         │        │
│  │    • Metrics: accuracy=0.72, f1=0.68, ...           │        │
│  │    • Artifacts: model, plots, ...                   │        │
│  │                                                      │        │
│  │  Run 2: lstm_direction_20240515_1415                │        │
│  │    • Params: hidden_size=128, lr=0.001, ...         │        │
│  │    • Metrics: accuracy=0.78, f1=0.75, ...           │        │
│  │    • Artifacts: model, plots, ...                   │        │
│  │                                                      │        │
│  │  ... (9 runs total)                                 │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              Model Registry                          │        │
│  │                                                      │        │
│  │  MarketPulsePredictor-direction                     │        │
│  │    • Version 1: rnn (Staging)                       │        │
│  │    • Version 2: lstm (Production) ⭐                │        │
│  │    • Version 3: gru (Staging)                       │        │
│  │                                                      │        │
│  │  MarketPulsePredictor-return                        │        │
│  │    • Version 1: lstm (Production) ⭐                │        │
│  │                                                      │        │
│  │  MarketPulsePredictor-volatility                    │        │
│  │    • Version 1: gru (Production) ⭐                 │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Load Production Models
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              FastAPI Server                          │        │
│  │                                                      │        │
│  │  On startup:                                         │        │
│  │    1. Connect to MLflow                              │        │
│  │    2. Load Production models for each task           │        │
│  │    3. Cache in memory                                │        │
│  │                                                      │        │
│  │  POST /predict:                                      │        │
│  │    1. Load latest features for ticker                │        │
│  │    2. Run through all 3 models                       │        │
│  │    3. Return predictions                             │        │
│  └──────────────────────────────────────────────────────┘        │
│                           │                                        │
│                           ▼                                        │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              Streamlit Dashboard                     │        │
│  │                                                      │        │
│  │  • Live predictions                                  │        │
│  │  • Model comparison                                  │        │
│  │  • Experiment history                                │        │
│  │  • Performance metrics                               │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Data → Sentiment → Features → Training → Models → Inference
   │          │           │          │         │         │
   │          │           │          │         │         └─▶ Predictions
   │          │           │          │         └───────────▶ Model Registry
   │          │           │          └─────────────────────▶ MLflow Experiments
   │          │           └────────────────────────────────▶ DVC Tracked
   │          └────────────────────────────────────────────▶ DVC Tracked
   └───────────────────────────────────────────────────────▶ DVC Tracked
```

## Training Timeline

```
Hour 0:00 ─┬─ Start setup_remote_training.sh
            │  • Package features
            │  • Start MLflow
            │  • Start ngrok
            └─▶ Get public URL

Hour 0:05 ─┬─ Upload to Google Drive
            └─▶ Open Colab notebook

Hour 0:10 ─┬─ Run Colab cells
            │  • Mount Drive
            │  • Clone repo
            │  • Extract features
            └─▶ Start training loop

Hour 0:15 ─┬─ Training begins (GPU)
            │
            ├─ RNN models (3 tasks)     [15-20 min]
            │  ├─ direction
            │  ├─ return
            │  └─ volatility
            │
            ├─ LSTM models (3 tasks)    [20-30 min]
            │  ├─ direction
            │  ├─ return
            │  └─ volatility
            │
            └─ GRU models (3 tasks)     [20-30 min]
               ├─ direction
               ├─ return
               └─ volatility

Hour 1:30 ─┬─ Training complete
            │  • All 9 models trained
            │  • All experiments logged
            │  • Models registered
            └─▶ Download trained_models.zip

Hour 1:35 ─┬─ Extract models locally
            └─▶ Verify in MLflow UI

Hour 1:40 ─┬─ Test API
            └─▶ Ready for deployment! 🎉
```

## Resource Usage

### Local Machine
- **CPU:** Minimal (just MLflow server)
- **RAM:** ~500 MB (MLflow)
- **Disk:** ~100 MB (feature data)
- **Network:** ~50 MB upload, ~100 MB download

### Google Colab
- **GPU:** T4 (16 GB VRAM) - fully utilized
- **RAM:** ~8-10 GB used (out of 12 GB available)
- **Disk:** ~500 MB (repo + features + models)
- **Time:** 1-2 hours

### MLflow Server
- **CPU:** <5% (logging only)
- **RAM:** ~200 MB
- **Disk:** ~50 MB (metadata)
- **Network:** ~10 MB (experiment logs)

## Cost Analysis

### Option 1: Colab + ngrok (Recommended)
- **Colab:** $0 (free tier)
- **ngrok:** $0 (free tier, 2-hour sessions)
- **MLflow:** $0 (local)
- **Total:** $0/month

### Option 2: Colab + Cloud MLflow
- **Colab:** $0 (free tier)
- **Railway/Render:** $5-10/month
- **PostgreSQL:** Included
- **Total:** $5-10/month

### Option 3: AWS SageMaker (Alternative)
- **ml.p3.2xlarge:** $3.06/hour
- **Training time:** 1-2 hours
- **Total:** $3-6 per training run

## Security Considerations

```
┌─────────────────────────────────────────────────────────┐
│                    Security Layers                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Local MLflow                                        │
│     • No authentication (localhost only)                │
│     • Firewall protected                                │
│                                                         │
│  2. ngrok Tunnel                                        │
│     • HTTPS encrypted                                   │
│     • Temporary URL (expires after 2 hours)             │
│     • No permanent exposure                             │
│                                                         │
│  3. Colab Environment                                   │
│     • Isolated VM                                       │
│     • No persistent storage                             │
│     • Automatic cleanup after session                   │
│                                                         │
│  4. Google Drive                                        │
│     • Private to your account                           │
│     • Encrypted at rest                                 │
│     • Access controlled                                 │
│                                                         │
│  5. Model Artifacts                                     │
│     • No sensitive data in models                       │
│     • Only learned weights                              │
│     • Safe to share/deploy                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting Flow

```
Problem: Training fails
    │
    ├─▶ Check: Feature data exists?
    │   └─▶ No: Run feature pipeline
    │   └─▶ Yes: Continue
    │
    ├─▶ Check: GPU available in Colab?
    │   └─▶ No: Runtime → Change runtime type → GPU
    │   └─▶ Yes: Continue
    │
    ├─▶ Check: MLflow connection?
    │   └─▶ No: Verify ngrok URL, restart if needed
    │   └─▶ Yes: Continue
    │
    ├─▶ Check: Out of memory?
    │   └─▶ Yes: Reduce batch_size to 32 or 16
    │   └─▶ No: Continue
    │
    └─▶ Check Colab logs for specific error
```

---

This architecture ensures:
- ✅ **Separation of concerns:** Data prep local, training remote
- ✅ **Cost efficiency:** Free GPU, minimal cloud costs
- ✅ **Reproducibility:** Full MLflow tracking
- ✅ **Scalability:** Easy to add more models/tasks
- ✅ **Security:** No permanent exposure, encrypted transfer
- ✅ **Flexibility:** Can switch to cloud MLflow anytime
