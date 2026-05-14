from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, description="Stock ticker symbol")
    feature_override: Optional[list[float]] = Field(None, description="Pre-built feature vector")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper()


class PredictionResponse(BaseModel):
    ticker: str
    direction: str = Field(..., pattern="^(UP|DOWN|FLAT)$")
    direction_confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_return: float
    volatility_spike: bool
    volatility_confidence: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    timestamp: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticker": "AAPL",
                "direction": "UP",
                "direction_confidence": 0.75,
                "predicted_return": 0.012,
                "volatility_spike": False,
                "volatility_confidence": 0.85,
                "model_name": "lstm",
                "timestamp": "2024-01-15T14:30:00Z",
            }
        }
    }


class ModelComparisonResponse(BaseModel):
    models: list[dict]

    model_config = {
        "json_schema_extra": {
            "example": {
                "models": [
                    {
                        "model_name": "lstm",
                        "task": "direction",
                        "accuracy": 0.72,
                        "f1": 0.68,
                        "rmse": None,
                        "run_id": "abc123",
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_data_ingestion: Optional[str] = None
