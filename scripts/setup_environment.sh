#!/bin/bash
set -e

echo "=========================================="
echo "Market Pulse Predictor - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ ! "$PYTHON_VERSION" =~ ^3\.11 ]]; then
    echo -e "${RED}Error: Python 3.11 is required. Found: $PYTHON_VERSION${NC}"
    echo "Please install Python 3.11 and try again."
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ./$VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.11 virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install requirements
echo "Installing packages from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All packages installed successfully${NC}"
else
    echo -e "${RED}✗ Package installation failed${NC}"
    exit 1
fi
echo ""

# Download NLTK data
echo "Downloading NLTK vader_lexicon data..."
python3 << EOF
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon', quiet=True)
print("✓ NLTK vader_lexicon downloaded")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ NLTK data downloaded${NC}"
else
    echo -e "${RED}✗ NLTK data download failed${NC}"
    exit 1
fi
echo ""

# Verify imports
echo "Verifying all imports..."
python3 << EOF
import sys

modules_to_test = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'numpy',
    'pandas',
    'pyarrow',
    'torch',
    'sklearn',
    'mlflow',
    'dvc',
    'dotenv',
    'yfinance',
    'requests_cache',
    'requests_ratelimiter',
    'feedparser',
    'praw',
    'httpx',
    'nltk',
    'transformers',
    'streamlit',
    'plotly',
    'airflow',
    'psycopg2'
]

failed_imports = []
for module in modules_to_test:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError as e:
        print(f"  ✗ {module}: {e}")
        failed_imports.append(module)

if failed_imports:
    print(f"\nFailed to import: {', '.join(failed_imports)}")
    sys.exit(1)
else:
    print("\n✓ All imports successful")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All imports verified${NC}"
else
    echo -e "${RED}✗ Some imports failed${NC}"
    exit 1
fi
echo ""

# Print summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Python version: $PYTHON_VERSION"
echo "  • Virtual environment: ./$VENV_DIR"
echo "  • Packages installed: $(pip list --format=freeze | wc -l | xargs)"
echo "  • NLTK data: vader_lexicon"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest tests/"
echo "  2. Start API: uvicorn src.api.main:app --reload"
echo "  3. Start Dashboard: streamlit run frontend/dashboard.py"
echo ""
