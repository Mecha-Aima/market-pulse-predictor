#!/bin/bash
# Setup script for market-pulse-predictor environment

set -e

echo "=== Market Pulse Predictor Environment Setup ==="

# Check Python version
echo "Checking Python version..."
python_version=$(python3.11 --version 2>&1 | awk '{print $2}')
if [[ $python_version < "3.11" ]]; then
    echo "Error: Python 3.11 or higher required"
    exit 1
fi
echo "✓ Python $python_version found"

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
pip install -r requirements-dev.txt --quiet
echo "✓ Dependencies installed"

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
echo "✓ NLTK data downloaded"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created - PLEASE FILL IN YOUR API KEYS"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw data/processed data/features models
mkdir -p data/raw/yahoo data/raw/news_rss data/raw/reddit data/raw/stocktwits data/raw/alphavantage
echo "✓ Directories created"

# Generate Airflow keys if not in .env
if ! grep -q "AIRFLOW__CORE__FERNET_KEY=." .env; then
    echo "Generating Airflow keys..."
    fernet_key=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    secret_key=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    
    # Update .env file
    sed -i.bak "s/AIRFLOW__CORE__FERNET_KEY=/AIRFLOW__CORE__FERNET_KEY=$fernet_key/" .env
    sed -i.bak "s/AIRFLOW__WEBSERVER__SECRET_KEY=/AIRFLOW__WEBSERVER__SECRET_KEY=$secret_key/" .env
    rm .env.bak
    echo "✓ Airflow keys generated"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Fill in API keys in .env file:"
echo "   - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD"
echo "   - ALPHAVANTAGE_API_KEY"
echo "   - AWS credentials (optional for local dev)"
echo "3. Run tests to verify setup: pytest tests/ -v"
echo "4. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000"
echo ""
