# Implementation Plan: Phases 5-7

## Overview
This plan implements REST API (Phase 5), Dashboard (Phase 6), and Airflow Orchestration (Phase 7) following TDD with real integration tests. No mocks for system behavior - only for external APIs.

## Pre-Implementation: Environment Setup

### Step 0.1: Install All Dependencies
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download NLTK data for VADER
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Step 0.2: Verify Phases 0-4
Run existing tests to confirm phases 0-4 are functional:
```bash
pytest tests/ -v --tb=short
```

If tests fail, fix them before proceeding. The progress report indicates tests are mostly mocks - we'll rewrite them to test real behavior.

### Step 0.3: Setup Local Services
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &

# Create necessary directories
mkdir -p models data/raw data/processed data/features
```

---

## Phase 5: REST API (TDD Approach)

### 5.1: Write API Tests FIRST (test_api.py)

**Test File:** `tests/test_api.py`

Tests to write:
1. `test_health_endpoint_returns_200_with_model_status`
   - Call GET /health
   - Assert 200 status
   - Assert response contains `status`, `model_loaded`, `last_data_ingestion`
   - Verify actual model loading state (not mocked)

2. `test_predict_endpoint_with_real_feature_data`
   - Create real feature data in `data/features/`
   - POST /predict with valid ticker
   - Assert response matches PredictionResponse schema
   - Verify predictions are numeric and within expected ranges

3. `test_predict_endpoint_unknown_ticker_404`
   - POST /predict with ticker="INVALID"
   - Assert 404 status
   - Assert error message contains ticker name

4. `test_predict_endpoint_no_models_loaded_503`
   - Remove models directory temporarily
   - POST /predict
   - Assert 503 status
   - Restore models directory

5. `test_tickers_endpoint_returns_configured_list`
   - GET /tickers
   - Assert response matches TARGET_TICKERS from env

6. `test_results_endpoint_returns_mlflow_metrics`
   - Requires MLflow server running
   - Create a test MLflow run with metrics
   - GET /results
   - Assert response contains run metrics

7. `test_cors_middleware_present`
   - Make OPTIONS request
   - Assert CORS headers present

### 5.2: Implement API Components

**Implementation Order:**

1. **`src/api/schemas.py`** - Pydantic models
   - Use Pydantic v2 syntax
   - Add field validation (e.g., ticker must be uppercase, 1-5 chars)
   - Add examples for documentation

2. **`src/api/model_loader.py`** - ModelRegistry class
   - Implement MLflow model loading with fallback to .pt files
   - Cache models in memory
   - Handle model loading errors gracefully
   - Log model loading events

3. **`src/api/main.py`** - FastAPI application
   - Implement all 7 routes from spec
   - Add CORS middleware
   - Add error handlers for 404, 422, 503
   - Add startup event to load models
   - Add lifespan context manager for cleanup

### 5.3: Integration Test

**Test File:** `tests/test_api_integration.py`

```python
@pytest.mark.integration
def test_full_prediction_pipeline():
    """
    End-to-end test:
    1. Start API server
    2. Load real trained model
    3. Create feature data
    4. Make prediction request
    5. Verify response
    """
    # This test uses real models and data
    pass
```

### 5.4: Manual Verification
```bash
# Start API server
uvicorn src.api.main:app --reload

# Test endpoints
curl http://localhost:8000/api/v1/health
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

---

## Phase 6: Frontend Dashboard (TDD Approach)

### 6.1: Write Dashboard Tests FIRST

**Test File:** `tests/test_dashboard.py`

Tests to write:
1. `test_dashboard_imports_successfully`
   - Import dashboard module
   - Assert no import errors

2. `test_api_client_handles_connection_errors`
   - Mock API to return connection error
   - Call API client function
   - Assert graceful error handling

3. `test_ticker_selection_widget_renders`
   - Use Streamlit testing framework
   - Assert ticker dropdown contains expected tickers

4. `test_prediction_display_formats_correctly`
   - Mock API response
   - Call display function
   - Assert output contains expected elements

### 6.2: Implement Dashboard Components

**Implementation Order:**

1. **`frontend/dashboard.py`** - Main dashboard
   - Section 1: Live Predictions
     - Ticker dropdown
     - Predict button
     - Result cards (direction, return, volatility)
     - Auto-refresh timer
   
   - Section 2: Sentiment Feed
     - Ticker selection
     - Bar chart (Plotly)
     - Data table
   
   - Section 3: Price Chart
     - Candlestick chart (Plotly)
     - Sentiment overlay
   
   - Section 4: Model Comparison
     - Comparison table
     - Loss curves
   
   - Section 5: Pipeline Status
     - Airflow DAG status
     - DVC pipeline log

2. **API Client Module** - `frontend/api_client.py`
   - Wrapper functions for all API calls
   - Error handling
   - Caching with `st.cache_data`

### 6.3: Manual Verification
```bash
# Start dashboard
streamlit run frontend/dashboard.py

# Open browser to http://localhost:8501
# Test all 5 sections
```

---

## Phase 7: Airflow Orchestration (TDD Approach)

### 7.1: Write DAG Tests FIRST

**Test File:** `tests/test_dags.py`

Tests to write:
1. `test_ingestion_dag_structure`
   - Load ingestion_dag
   - Assert 5 tasks exist
   - Assert task dependencies correct (1-4 parallel, 5 after all)
   - Assert schedule is `*/30 * * * *`

2. `test_sentiment_dag_structure`
   - Load sentiment_dag
   - Assert 2 tasks exist
   - Assert schedule is `5 * * * *`

3. `test_feature_dag_structure`
   - Load feature_dag
   - Assert 2 tasks exist
   - Assert schedule is `15 * * * *`

4. `test_training_dag_structure`
   - Load training_dag
   - Assert 4 tasks exist
   - Assert tasks 1-3 parallel, 4 after all
   - Assert schedule is `0 2 * * *`

5. `test_ingestion_dag_execution`
   - Use Airflow testing utilities
   - Run ingestion_dag with test data
   - Assert data files created
   - Assert DVC tracking updated

### 7.2: Implement DAGs

**Implementation Order:**

1. **`airflow/dags/ingestion_dag.py`**
   - Import scrapers from src/
   - Create PythonOperators for each scraper
   - Create BashOperator for DVC add
   - Set up parallel execution for scrapers
   - Add error handling and retries

2. **`airflow/dags/sentiment_dag.py`**
   - Import sentiment analyzer
   - Create PythonOperator for sentiment analysis
   - Create BashOperator for DVC add
   - Add dependency on ingestion_dag

3. **`airflow/dags/feature_dag.py`**
   - Import feature builder
   - Create PythonOperator for feature building
   - Create BashOperator for DVC add
   - Add dependency on sentiment_dag

4. **`airflow/dags/training_dag.py`**
   - Import training modules
   - Create PythonOperators for each model
   - Create PythonOperator for evaluation
   - Set up parallel execution for training
   - Add MLflow logging

### 7.3: Setup Airflow Environment

```bash
# Initialize Airflow database
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Airflow services
airflow webserver --port 8080 &
airflow scheduler &
```

### 7.4: Manual Verification
```bash
# Open Airflow UI
# http://localhost:8080

# Trigger each DAG manually
# Verify task execution
# Check logs for errors
```

---

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external APIs (Reddit, Alpha Vantage, Yahoo Finance)
- Do NOT mock internal modules

### Integration Tests
- Test multiple components together
- Use real data files
- Use real models
- Use real MLflow server
- Tag with `@pytest.mark.integration`

### End-to-End Tests
- Test complete workflows
- Start services (API, MLflow)
- Run full pipeline
- Verify outputs

### Coverage Requirements
- Overall: >= 70%
- src/api/: >= 90%
- src/models/: >= 85%
- src/sentiment/: >= 80%

---

## Implementation Checklist

### Phase 5: REST API
- [ ] Write all API tests
- [ ] Implement schemas.py
- [ ] Implement model_loader.py
- [ ] Implement main.py with all routes
- [ ] Run tests: `pytest tests/test_api.py -v`
- [ ] Manual verification with curl
- [ ] Integration test with real models

### Phase 6: Dashboard
- [ ] Write dashboard tests
- [ ] Implement Section 1: Live Predictions
- [ ] Implement Section 2: Sentiment Feed
- [ ] Implement Section 3: Price Chart
- [ ] Implement Section 4: Model Comparison
- [ ] Implement Section 5: Pipeline Status
- [ ] Run tests: `pytest tests/test_dashboard.py -v`
- [ ] Manual verification in browser

### Phase 7: Airflow
- [ ] Write DAG tests
- [ ] Implement ingestion_dag.py
- [ ] Implement sentiment_dag.py
- [ ] Implement feature_dag.py
- [ ] Implement training_dag.py
- [ ] Run tests: `pytest tests/test_dags.py -v`
- [ ] Setup Airflow environment
- [ ] Manual verification in Airflow UI

### Final Verification
- [ ] Run full test suite: `pytest tests/ -v --cov=src`
- [ ] Verify coverage >= 70%
- [ ] Run end-to-end pipeline
- [ ] Document any issues in progress.md

---

## Notes

1. **No Mocking Internal Behavior**: Tests should use real models, real data files, and real services. Only mock external APIs.

2. **Real Dependencies**: All packages must be installed. No import guards or try/except for missing packages.

3. **TDD Discipline**: Write tests first, see them fail, then implement to make them pass.

4. **Incremental Development**: Complete one phase fully before moving to the next.

5. **Documentation**: Update progress.md after each phase completion.
