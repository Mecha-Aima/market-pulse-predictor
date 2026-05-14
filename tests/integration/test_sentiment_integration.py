"""
Integration tests for sentiment analysis pipeline.

Tests sentiment analysis using real VADER and actual fixture data.
"""

import pandas as pd
import pytest

from src.sentiment.analyzer import SentimentAnalyzer


@pytest.mark.integration
class TestSentimentAnalysisIntegration:
    """Integration tests for sentiment analyzer with real VADER."""

    def test_analyze_positive_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "Apple stock surges to new all-time high on strong earnings report!"
        )
        assert "label" in result
        assert "score" in result
        assert result["label"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
        assert -1 <= result["score"] <= 1
        assert result["label"] == "POSITIVE"

    def test_analyze_negative_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Market crash fears grow as recession looms.")
        assert result["label"] == "NEGATIVE"

    def test_analyze_neutral_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The stock price remained unchanged today.")
        assert result["label"] == "NEUTRAL"

    def test_batch_analyze_real_news_fixture(self, fixture_news_data):
        analyzer = SentimentAnalyzer()
        records = fixture_news_data[:5]
        results = analyzer.batch_analyze(records)

        assert len(results) == len(records)
        for r in results:
            assert "sentiment_label" in r
            assert "sentiment_score" in r
            assert "text" in r
            assert "ticker" in r

    def test_batch_analyze_skips_none_text(self):
        analyzer = SentimentAnalyzer()
        data = [
            {"ticker": "AAPL", "text": "Strong earnings beat expectations."},
            {"ticker": "AAPL", "text": None},
            {"ticker": "AAPL", "text": "Company faces major losses."},
        ]
        results = analyzer.batch_analyze(data)

        assert len(results) == 3
        assert results[0]["sentiment_label"] == "POSITIVE"
        assert results[1]["sentiment_label"] is None
        assert results[1]["sentiment_score"] is None
        assert results[2]["sentiment_label"] == "NEGATIVE"


@pytest.mark.integration
class TestSentimentPipeline:
    """Integration tests for complete sentiment pipeline."""

    def test_batch_analyze_preserves_original_fields(self):
        analyzer = SentimentAnalyzer()
        data = [
            {
                "ticker": "AAPL",
                "text": "Apple announces record-breaking quarterly revenue.",
                "source": "test",
            },
            {
                "ticker": "AAPL",
                "text": "Concerns raised about supply chain disruptions.",
                "source": "test",
            },
        ]
        results = analyzer.batch_analyze(data)

        assert len(results) == len(data)
        for r in results:
            assert r["ticker"] == "AAPL"
            assert r["source"] == "test"
            assert "sentiment_label" in r
            assert r["sentiment_label"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")

    def test_batch_analyze_writes_parquet(self, tmp_path):
        analyzer = SentimentAnalyzer()
        data = [
            {"ticker": "AAPL", "text": "Great earnings!"},
            {"ticker": "AAPL", "text": "Terrible quarter."},
        ]
        results = analyzer.batch_analyze(data)
        df = pd.DataFrame(results)

        out = tmp_path / "sentiment.parquet"
        df.to_parquet(out, index=False)

        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 2
        assert "sentiment_label" in loaded.columns

    def test_alphavantage_sentiment_preserved(self):
        """batch_analyze uses av_sentiment_label when present instead of running VADER."""
        analyzer = SentimentAnalyzer()
        data = [
            {
                "ticker": "AAPL",
                "text": "Some news text",
                "av_sentiment_label": "Bullish",
                "av_sentiment_score": 0.75,
            }
        ]
        results = analyzer.batch_analyze(data)

        assert results[0]["sentiment_label"] == "POSITIVE"
        assert results[0]["av_sentiment_score"] == 0.75

    def test_fixture_sentiment_data_structure(self, fixture_sentiment_data):
        """Verify the sentiment fixture matches the expected schema."""
        df = fixture_sentiment_data
        assert "vader_compound" in df.columns
        assert "ticker" in df.columns
        assert len(df) > 0
        assert df["vader_compound"].between(-1, 1).all()
