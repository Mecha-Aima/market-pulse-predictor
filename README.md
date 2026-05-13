# market-pulse-predictor

Real-time market movement prediction pipeline covering ingestion, sentiment analysis,
time-series feature generation, and sequential model training.

## 🚀 Quick Start

### 1. Data Collection & Feature Engineering (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run data pipeline
python -m src.ingestion.run_all --lookback-hours 168
python -m src.sentiment.run_sentiment
python -m src.features.run_features
```

### 2. Model Training (Google Colab - GPU)

**Why Colab?** Training RNN/LSTM/GRU models requires GPU. Colab provides free T4 GPU access.

```bash
# Prepare data for Colab
./scripts/setup_remote_training.sh

# Follow the prompts to:
# 1. Package feature data
# 2. Start MLflow server (optional)
# 3. Create ngrok tunnel (optional)
```

Then:
1. Upload `colab_upload/features.tar.gz` to Google Drive
2. Open `notebooks/colab_training.ipynb` in Colab
3. Run all cells (trains 9 models in ~1-2 hours)
4. Download trained models back to local `models/` directory

**📖 Training Documentation:**
- **[TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)** - Start here! Answers all your questions
- **[QUICK_START_TRAINING.md](QUICK_START_TRAINING.md)** - 15-minute quick start guide
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide
- **[TRAINING_OPTIONS.md](TRAINING_OPTIONS.md)** - Compare all training options
- **[docs/training_architecture.md](docs/training_architecture.md)** - Visual architecture diagrams

### 3. Run API & Dashboard (Local)
```bash
# Start API
uvicorn src.api.main:app --reload

# In another terminal, start dashboard
streamlit run frontend/dashboard.py
```

Access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5001

## 📊 Architecture

```
Data Sources → Ingestion → Sentiment → Features → Training (Colab GPU)
                                                        ↓
                                                   MLflow Tracking
                                                        ↓
                                              Model Registry
                                                        ↓
                                              FastAPI + Streamlit
```

## 🎯 Features

- **Multi-source data ingestion:** Yahoo Finance, Alpha Vantage, Reddit, StockTwits
- **Sentiment analysis:** VADER + optional FinBERT
- **3 model architectures:** SimpleRNN, LSTM, GRU
- **3 prediction tasks:** Direction, Return, Volatility
- **MLflow experiment tracking:** Full reproducibility
- **DVC data versioning:** Track data lineage
- **REST API:** FastAPI with Pydantic validation
- **Interactive dashboard:** Streamlit with real-time updates

## 📝 Notes

- Yahoo Finance ingestion is intentionally rate-limited and cached to reduce breakage risk.
- Reddit uses script auth and now expects username/password in addition to app credentials.
- StockTwits is the unauthenticated social-text source.
- Alpha Vantage is the primary structured news API and can provide pre-computed sentiment.

## 🔧 Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=src

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## 📦 Deployment

See [Phase 9 Deployment Guide](.kiro/specs/phases-8-9-deployment/) for:
- Docker containerization
- AWS EC2 deployment
- CI/CD with GitHub Actions
- Production MLflow setup

