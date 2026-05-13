#!/bin/bash
# Setup script for remote training on Colab

set -e

echo "🚀 Market Pulse Predictor - Remote Training Setup"
echo "=================================================="

# Step 1: Check if data exists
if [ ! -d "data/features" ] || [ -z "$(ls -A data/features)" ]; then
    echo "❌ Error: No feature data found in data/features/"
    echo "   Run the feature pipeline first: python -m src.features.run_features"
    exit 1
fi

echo "✅ Feature data found"

# Step 2: Package data for upload
echo "📦 Packaging data for Colab..."
mkdir -p colab_upload
tar -czf colab_upload/features.tar.gz data/features/
echo "✅ Created colab_upload/features.tar.gz"

# Step 3: Start local MLflow server (optional)
read -p "Do you want to start a local MLflow server? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔧 Starting MLflow server on port 5001..."
    mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db &
    MLFLOW_PID=$!
    echo "✅ MLflow server started (PID: $MLFLOW_PID)"
    echo "   Access at: http://localhost:5001"
    
    # Step 4: Start ngrok (if installed)
    if command -v ngrok &> /dev/null; then
        read -p "Do you want to expose MLflow via ngrok? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🌐 Starting ngrok tunnel..."
            ngrok http 5001 > /dev/null &
            NGROK_PID=$!
            sleep 3
            NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*' | head -1)
            echo "✅ Ngrok tunnel created: $NGROK_URL"
            echo "   Use this URL in your Colab notebook!"
            echo "$NGROK_URL" > colab_upload/mlflow_url.txt
        fi
    else
        echo "⚠️  ngrok not found. Install it from: https://ngrok.com/download"
        echo "   Or use a cloud MLflow deployment instead."
    fi
fi

# Step 5: Instructions
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Upload colab_upload/features.tar.gz to Google Drive"
echo "2. Open notebooks/colab_training.ipynb in Colab"
echo "3. Update the MLflow tracking URI in the notebook"
if [ -f "colab_upload/mlflow_url.txt" ]; then
    echo "   Use this URL: $(cat colab_upload/mlflow_url.txt)"
fi
echo "4. Run all cells to train models"
echo "5. Download trained models back to local models/ directory"
echo ""
echo "✨ Setup complete!"
