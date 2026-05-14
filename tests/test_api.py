
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def test_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("TARGET_TICKERS", "AAPL,MSFT,GOOGL")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MODEL_REGISTRY_PATH", "./models/")


@pytest.mark.asyncio
async def test_health_endpoint_returns_200_with_model_status(test_env):
    """Test health endpoint returns 200 with model status"""
    from src.api.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "last_data_ingestion" in data
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_endpoint_unknown_ticker_404(test_env):
    """Test predict endpoint returns 404 for unknown ticker"""
    from src.api.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"ticker": "FAKE"}
        )
    
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "FAKE" in str(data["detail"])


@pytest.mark.asyncio
async def test_tickers_endpoint_returns_configured_list(test_env):
    """Test tickers endpoint returns configured list"""
    from src.api.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/tickers")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "AAPL" in data
    assert "MSFT" in data
    assert "GOOGL" in data


@pytest.mark.asyncio
async def test_cors_middleware_present(test_env):
    """Test CORS headers are present"""
    from src.api.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"}
        )
    
    # CORS middleware should add these headers
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_results_endpoint_returns_list(test_env):
    """Test results endpoint returns list of model metrics"""
    from src.api.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/results")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
