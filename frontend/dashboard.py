import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.api_client import APIClient

# Page config
st.set_page_config(
    page_title="Market Pulse Predictor",
    page_icon="📈",
    layout="wide"
)

# Initialize API client
client = APIClient()

# Sidebar navigation
st.sidebar.title("Market Pulse Predictor")
section = st.sidebar.radio(
    "Navigation",
    ["Live Predictions", "Sentiment Feed", "Price Chart", "Model Comparison", "Pipeline Status"]
)

# Section 1: Live Predictions
if section == "Live Predictions":
    st.title("📈 Live Market Predictions")
    
    # Get tickers
    tickers = client.get_tickers()
    
    if not tickers:
        st.error("Failed to load tickers. Please check API connection.")
    else:
        # Ticker selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_ticker = st.selectbox("Select Ticker", tickers)
        with col2:
            predict_button = st.button("Predict Now", type="primary")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (every 5 minutes)")
        
        if predict_button or auto_refresh:
            with st.spinner("Fetching prediction..."):
                prediction = client.predict(selected_ticker)
            
            if prediction:
                # Display results in three columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Market Direction",
                        value=prediction["direction"],
                        delta=f"{prediction['direction_confidence']:.1%} confidence"
                    )
                    # Color-coded direction
                    if prediction["direction"] == "UP":
                        st.success("📈 Bullish Signal")
                    elif prediction["direction"] == "DOWN":
                        st.error("📉 Bearish Signal")
                    else:
                        st.info("➡️ Neutral Signal")
                
                with col2:
                    st.metric(
                        label="Predicted Return",
                        value=f"{prediction['predicted_return']:.2%}",
                        delta="Expected price movement"
                    )
                
                with col3:
                    spike_text = "YES" if prediction["volatility_spike"] else "NO"
                    st.metric(
                        label="Volatility Spike",
                        value=spike_text,
                        delta=f"{prediction['volatility_confidence']:.1%} confidence"
                    )
                
                # Metadata
                model_name = prediction["model_name"]
                ts = prediction["timestamp"]
                st.caption(f"Model: {model_name} | Timestamp: {ts}")
        
        # Auto-refresh countdown
        if auto_refresh:
            placeholder = st.empty()
            for remaining in range(300, 0, -1):
                mins, secs = divmod(remaining, 60)
                placeholder.text(f"Next refresh in: {mins:02d}:{secs:02d}")
                time.sleep(1)
            st.rerun()

# Section 2: Sentiment Feed
elif section == "Sentiment Feed":
    st.title("💬 Sentiment Analysis Feed")
    
    tickers = client.get_tickers()
    if tickers:
        selected_ticker = st.selectbox("Select Ticker", tickers)
        
        sentiment_data = client.get_sentiment(selected_ticker)
        
        if sentiment_data and sentiment_data.get("data"):
            st.info("Sentiment data visualization will be implemented when data is available")
            st.json(sentiment_data)
        else:
            st.warning(f"No sentiment data available for {selected_ticker}")

# Section 3: Price Chart
elif section == "Price Chart":
    st.title("📊 Price Chart")
    
    tickers = client.get_tickers()
    if tickers:
        selected_ticker = st.selectbox("Select Ticker", tickers)
        
        price_data = client.get_prices(selected_ticker)
        
        if price_data and price_data.get("data"):
            st.info("Price chart will be implemented when data is available")
            st.json(price_data)
        else:
            st.warning(f"No price data available for {selected_ticker}")

# Section 4: Model Comparison
elif section == "Model Comparison":
    st.title("🤖 Model Comparison")
    
    results = client.get_results()
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No model comparison data available yet. Train models to see results.")

# Section 5: Pipeline Status
elif section == "Pipeline Status":
    st.title("⚙️ Pipeline Status")
    
    health = client.get_health()
    
    if health:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if health["status"] == "ok" else "🔴"
            st.metric("API Status", f"{status_color} {health['status'].upper()}")
        
        with col2:
            model_status = "🟢 Loaded" if health["model_loaded"] else "🔴 Not Loaded"
            st.metric("Models", model_status)
        
        with col3:
            last_ingest = health.get("last_data_ingestion", "Never")
            if last_ingest != "Never":
                last_ingest = last_ingest[:19]  # Trim to readable format
            st.metric("Last Data Ingestion", last_ingest)
    else:
        st.error("Failed to fetch pipeline status")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Market Pulse Predictor v1.0")
