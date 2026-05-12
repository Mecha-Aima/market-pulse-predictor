import json
from datetime import datetime, timezone

import pandas as pd

from src.sentiment.analyzer import SentimentAnalyzer


def make_record(text: str | None) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": "record-1",
        "source": "reddit",
        "ticker": "AAPL",
        "timestamp": timestamp,
        "text": text,
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
        "score": None,
        "url": None,
    }


def test_vader_positive_text() -> None:
    result = SentimentAnalyzer().analyze("Stock market rally! Strong earnings beat expectations.")

    assert result["label"] == "POSITIVE"


def test_vader_negative_text() -> None:
    result = SentimentAnalyzer().analyze("Market crash fears grow as recession looms.")

    assert result["label"] == "NEGATIVE"


def test_vader_neutral_text() -> None:
    result = SentimentAnalyzer().analyze("The market opened today.")

    assert result["label"] == "NEUTRAL"


def test_batch_analyze_skips_none_text() -> None:
    records = SentimentAnalyzer().batch_analyze([make_record(None)])

    assert records[0]["sentiment_label"] is None
    assert records[0]["sentiment_score"] is None


def test_batch_analyze_returns_enriched_records() -> None:
    records = SentimentAnalyzer().batch_analyze([make_record("AAPL looks strong today.")])

    assert "sentiment_label" in records[0]
    assert "sentiment_score" in records[0]


def test_batch_analyze_uses_alphavantage_sentiment_when_present() -> None:
    record = {
        **make_record("Apple rises on earnings."),
        "av_sentiment_label": "Bullish",
        "av_sentiment_score": 0.88,
    }

    records = SentimentAnalyzer().batch_analyze([record])

    assert records[0]["sentiment_label"] == "POSITIVE"
    assert records[0]["sentiment_score"] == 0.88


def test_run_sentiment_writes_parquet(tmp_path, monkeypatch) -> None:
    from src.sentiment import run_sentiment

    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data" / "raw" / "reddit"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "AAPL_20240115T140000Z.json"
    raw_path.write_text(json.dumps([make_record("AAPL is climbing quickly.")]))

    class FakeAnalyzer:
        def batch_analyze(self, records: list[dict]) -> list[dict]:
            enriched = []
            for record in records:
                enriched.append(
                    {
                        **record,
                        "sentiment_label": "POSITIVE",
                        "sentiment_score": 0.9,
                    }
                )
            return enriched

    monkeypatch.setattr(run_sentiment, "SentimentAnalyzer", FakeAnalyzer)

    output_paths = run_sentiment.process_unprocessed_files()

    assert len(output_paths) == 1
    output_path = output_paths[0]
    assert output_path.suffix == ".parquet"
    frame = pd.read_parquet(output_path)
    assert frame.loc[0, "sentiment_label"] == "POSITIVE"
