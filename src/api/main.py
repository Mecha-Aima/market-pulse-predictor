import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_loader import registry
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Load models
    logger.info("Starting up API server...")
    for task in ["direction", "return", "volatility"]:
        model = registry.get_model(task)
        if model:
            logger.info(f"Loaded model for task: {task}")

    yield

    # Shutdown
    logger.info("Shutting down API server...")


app = FastAPI(title="Market Pulse Predictor API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    # Check for last data ingestion
    last_ingestion = None
    raw_data_path = Path("data/raw")
    if raw_data_path.exists():
        files = list(raw_data_path.rglob("*.json"))
        if files:
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            last_ingestion = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

    return HealthResponse(
        status="ok", model_loaded=registry.models_loaded(), last_data_ingestion=last_ingestion
    )


@app.get("/api/v1/tickers", response_model=list[str])
async def get_tickers():
    """Get list of tracked tickers"""
    tickers_str = os.getenv("TARGET_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA,SPY")
    return tickers_str.split(",")


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run prediction for a ticker"""
    # Validate ticker is in tracked list
    tickers = await get_tickers()
    if request.ticker not in tickers:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {request.ticker} not tracked. Available tickers: {', '.join(tickers)}",
        )

    # Check if models are loaded
    direction_model = registry.get_model("direction")
    return_model = registry.get_model("return")
    volatility_model = registry.get_model("volatility")

    if not all([direction_model, return_model, volatility_model]):
        raise HTTPException(
            status_code=503, detail="Models not loaded. Please wait for model initialization."
        )

    # For now, return mock predictions
    # TODO: Implement actual prediction logic with feature loading
    return PredictionResponse(
        ticker=request.ticker,
        direction="UP",
        direction_confidence=0.75,
        predicted_return=0.012,
        volatility_spike=False,
        volatility_confidence=0.85,
        model_name="lstm",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/api/v1/results")
async def get_results():
    """Get latest model comparison metrics from MLflow"""
    # TODO: Implement MLflow metrics fetching
    return []


@app.get("/api/v1/experiments")
async def get_experiments():
    """List all MLflow runs with key metrics"""
    # TODO: Implement MLflow experiments listing
    return []


@app.get("/api/v1/sentiment/{ticker}")
async def get_sentiment(ticker: str):
    """Get last 24h of aggregated sentiment for a ticker"""
    tickers = await get_tickers()
    if ticker not in tickers:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    # TODO: Implement sentiment data fetching
    return {"ticker": ticker, "data": []}


@app.get("/api/v1/prices/{ticker}")
async def get_prices(ticker: str):
    """Get last 24h of OHLCV for a ticker"""
    tickers = await get_tickers()
    if ticker not in tickers:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    # TODO: Implement price data fetching
    return {"ticker": ticker, "data": []}
