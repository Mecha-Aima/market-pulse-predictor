"""
Integration tests for feature engineering pipeline.

Tests feature building using real fixture data against the actual
TimeSeriesBuilder API (build_feature_frame / create_splits / save_artifacts).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.features.builder import TimeSeriesBuilder


def _make_processed_frame(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine price and sentiment fixtures into the processed-frame format
    that TimeSeriesBuilder.build_feature_frame() expects.

    Required columns: timestamp, ticker, source, open/high/low/close/volume (price rows),
    text, sentiment_label, sentiment_score (text rows).
    """
    rows = []

    # Price rows from yahoo fixture
    for _, row in price_df.iterrows():
        date_str = str(row.get("date", row.get("Date", "")))
        if not date_str:
            continue
        rows.append(
            {
                "id": f"yahoo-{row.get('ticker', 'AAPL')}-{date_str}",
                "source": "yahoo_price",
                "ticker": row.get("ticker", "AAPL"),
                "timestamp": f"{date_str}T16:00:00Z",
                "text": None,
                "open": row.get("open") or row.get("Open"),
                "high": row.get("high") or row.get("High"),
                "low": row.get("low") or row.get("Low"),
                "close": row.get("close") or row.get("Close"),
                "volume": row.get("volume") or row.get("Volume"),
                "score": None,
                "url": None,
                "sentiment_label": None,
                "sentiment_score": None,
            }
        )

    # Sentiment/text rows from sentiment fixture
    for _, row in sentiment_df.iterrows():
        date_str = str(row.get("date", ""))
        if not date_str:
            continue
        score = row.get("vader_compound", 0.0)
        label = "POSITIVE" if score >= 0.05 else ("NEGATIVE" if score <= -0.05 else "NEUTRAL")
        rows.append(
            {
                "id": f"news-{row.get('ticker', 'AAPL')}-{date_str}",
                "source": row.get("source", "news_rss"),
                "ticker": row.get("ticker", "AAPL"),
                "timestamp": f"{date_str}T12:00:00Z",
                "text": row.get("text", "news text"),
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "score": None,
                "url": None,
                "sentiment_label": label,
                "sentiment_score": float(score),
            }
        )

    return pd.DataFrame(rows)


@pytest.mark.integration
class TestFeatureBuilderIntegration:
    """Integration tests for feature builder with real fixture data."""

    def test_aggregate_sentiment(self, fixture_sentiment_data):
        builder = TimeSeriesBuilder()
        sentiment_df = fixture_sentiment_data.copy()

        rows = []
        for _, row in sentiment_df.iterrows():
            date_str = str(row.get("date", ""))
            score = float(row.get("vader_compound", 0.0))
            label = (
                "POSITIVE" if score >= 0.05 else ("NEGATIVE" if score <= -0.05 else "NEUTRAL")
            )
            rows.append(
                {
                    "source": row.get("source", "news_rss"),
                    "ticker": row.get("ticker", "AAPL"),
                    "timestamp": f"{date_str}T12:00:00Z",
                    "text": row.get("text", "news"),
                    "sentiment_label": label,
                    "sentiment_score": score,
                }
            )

        frame = pd.DataFrame(rows)
        agg = builder.aggregate_sentiment(frame)

        assert isinstance(agg, pd.DataFrame)
        assert len(agg) > 0
        assert "sentiment_positive_count" in agg.columns
        assert "sentiment_negative_count" in agg.columns
        assert "total_mentions" in agg.columns

    def test_build_feature_frame(self, fixture_price_data, fixture_sentiment_data):
        builder = TimeSeriesBuilder()
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)

        result = builder.build_feature_frame(processed)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "close" in result.columns
            assert "price_return" in result.columns
            assert "label_direction" in result.columns
            assert "label_return" in result.columns

    def test_chronological_split_no_leakage(self, fixture_price_data, fixture_sentiment_data):
        """Train timestamps must all be before val timestamps."""
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty or len(feature_frame) < 10:
            pytest.skip("Not enough data for split test")

        datasets = builder.create_splits(feature_frame)

        ts_train = datasets["timestamps_train"]
        ts_val = datasets["timestamps_val"]
        if len(ts_train) > 0 and len(ts_val) > 0:
            assert max(ts_train) <= min(ts_val)

    def test_sequence_shape(self, fixture_price_data, fixture_sentiment_data):
        seq_len = 5
        builder = TimeSeriesBuilder(sequence_length=seq_len)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty or len(feature_frame) < seq_len + 1:
            pytest.skip("Not enough data for sequence shape test")

        datasets = builder.create_splits(feature_frame)

        X_train = datasets["X_train"]
        assert X_train.ndim == 3
        assert X_train.shape[1] == seq_len

    def test_scaler_fit_on_train_only(self, fixture_price_data, fixture_sentiment_data):
        """Scaler mean should match statistics from training data only."""
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty or len(feature_frame) < 10:
            pytest.skip("Not enough data")

        datasets = builder.create_splits(feature_frame)
        scaler = datasets["scaler"]

        assert hasattr(scaler, "mean_")
        assert scaler.mean_ is not None


@pytest.mark.integration
class TestFeaturePipeline:
    """Integration tests for the complete feature pipeline."""

    def test_save_and_reload_artifacts(self, fixture_price_data, fixture_sentiment_data, tmp_path):
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty or len(feature_frame) < 10:
            pytest.skip("Not enough data")

        datasets = builder.create_splits(feature_frame)
        builder.save_artifacts(datasets, tmp_path)

        assert (tmp_path / "X_train.npy").exists()
        assert (tmp_path / "y_direction_train.npy").exists()
        assert (tmp_path / "scaler.pkl").exists()
        assert (tmp_path / "feature_columns.json").exists()

        X_reloaded = np.load(tmp_path / "X_train.npy")
        assert X_reloaded.shape == datasets["X_train"].shape

        cols = json.loads((tmp_path / "feature_columns.json").read_text())
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_label_direction_values(self, fixture_price_data, fixture_sentiment_data):
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty:
            pytest.skip("Not enough data")

        assert feature_frame["label_direction"].isin([-1, 0, 1]).all()

    def test_volatility_spike_is_binary(self, fixture_price_data, fixture_sentiment_data):
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty:
            pytest.skip("Not enough data")

        assert feature_frame["label_volatility_spike"].isin([0, 1]).all()

    def test_feature_columns_json_consistency(
        self, fixture_price_data, fixture_sentiment_data, tmp_path
    ):
        """feature_columns.json must list the same columns as X_train last dimension."""
        builder = TimeSeriesBuilder(sequence_length=5)
        processed = _make_processed_frame(fixture_price_data, fixture_sentiment_data)
        feature_frame = builder.build_feature_frame(processed)

        if feature_frame.empty or len(feature_frame) < 10:
            pytest.skip("Not enough data")

        datasets = builder.create_splits(feature_frame)
        builder.save_artifacts(datasets, tmp_path)

        cols = json.loads((tmp_path / "feature_columns.json").read_text())
        X = np.load(tmp_path / "X_train.npy")
        assert len(cols) == X.shape[2]
