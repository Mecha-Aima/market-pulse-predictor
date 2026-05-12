from fastapi import FastAPI

from src.api.schemas import HealthResponse

app = FastAPI(title="market-pulse-predictor")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
