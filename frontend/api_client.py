import os

import httpx
import streamlit as st


class APIClient:
    """Client for interacting with the Market Pulse API"""

    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
        self.timeout = 10.0

    @st.cache_data(ttl=60)
    def get_health(_self):
        try:
            response = httpx.get(f"{_self.base_url}/health", timeout=_self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            return None

    @st.cache_data(ttl=60)
    def get_tickers(_self):
        try:
            response = httpx.get(f"{_self.base_url}/tickers", timeout=_self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch tickers: {e}")
            return []

    def predict(_self, ticker: str):
        try:
            response = httpx.post(
                f"{_self.base_url}/predict",
                json={"ticker": ticker},
                timeout=_self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                st.error(f"Ticker {ticker} not found")
            elif e.response.status_code == 503:
                st.error("Models not loaded yet. Please wait.")
            else:
                st.error(f"Prediction failed: {e}")
            return None
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None

    @st.cache_data(ttl=60)
    def get_results(_self):
        try:
            response = httpx.get(f"{_self.base_url}/results", timeout=_self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Failed to fetch results: {e}")
            return []

    @st.cache_data(ttl=60)
    def get_sentiment(_self, ticker: str):
        try:
            response = httpx.get(
                f"{_self.base_url}/sentiment/{ticker}", timeout=_self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Failed to fetch sentiment: {e}")
            return {"ticker": ticker, "data": []}

    @st.cache_data(ttl=60)
    def get_prices(_self, ticker: str):
        try:
            response = httpx.get(
                f"{_self.base_url}/prices/{ticker}", timeout=_self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Failed to fetch prices: {e}")
            return {"ticker": ticker, "data": []}
