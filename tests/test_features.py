from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.features.builder import TimeSeriesBuilder


def make_processed_record(
    *,
    timestamp: datetime,
    source: str,
    ticker: str = "AAPL",
    text: str | None = "text",
    sentiment_label: str | None = "POSITIVE",
    sentiment_score: float | None = 0.8,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
    volume: float | None = None,
) -> dict:
    return {
        "id": f"{source}-{timestamp.isoformat()}",
        "source": source,
        "ticker": ticker,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "text": text,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "score": None,
        "url": None,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
    }


def test_sentiment_aggregation_correct_counts() -> None:
    builder = TimeSeriesBuilder(sequence_length=3)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    records = [
        make_processed_record(timestamp=now, source="reddit", sentiment_label="POSITIVE"),
        make_processed_record(timestamp=now, source="reddit", sentiment_label="POSITIVE"),
        make_processed_record(timestamp=now, source="stocktwits", sentiment_label="POSITIVE"),
        make_processed_record(
            timestamp=now,
            source="news_rss",
            sentiment_label="NEGATIVE",
            sentiment_score=-0.7,
        ),
        make_processed_record(
            timestamp=now, source="yahoo_news", sentiment_label="NEGATIVE", sentiment_score=-0.2
        ),
    ]

    aggregated = builder.aggregate_sentiment(pd.DataFrame(records))

    assert aggregated.loc[0, "sentiment_positive_count"] == 3
    assert aggregated.loc[0, "sentiment_negative_count"] == 2


def test_feature_merge_no_price_no_row() -> None:
    builder = TimeSeriesBuilder(sequence_length=3)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    sentiment_frame = pd.DataFrame(
        [make_processed_record(timestamp=now, source="reddit", sentiment_label="POSITIVE")]
    )

    merged = builder.build_feature_frame(sentiment_frame)

    assert merged.empty


def test_label_direction_correct() -> None:
    builder = TimeSeriesBuilder(sequence_length=3)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    records = []
    for offset in range(4):
        timestamp = start + timedelta(hours=offset)
        records.append(
            make_processed_record(
                timestamp=timestamp,
                source="reddit",
                sentiment_label="POSITIVE",
                sentiment_score=0.6,
            )
        )
        records.append(
            make_processed_record(
                timestamp=start + timedelta(hours=offset),
                source="yahoo_price",
                text=None,
                sentiment_label=None,
                sentiment_score=None,
                open_price=100 + offset,
                high=101 + offset,
                low=99 + offset,
                close=100 + offset + (0.5 if offset == 1 else 0),
                volume=1000,
            )
        )

    feature_frame = builder.build_feature_frame(pd.DataFrame(records))

    assert 1 in feature_frame["label_direction"].tolist()


def test_volatility_spike_detection() -> None:
    builder = TimeSeriesBuilder(sequence_length=3)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    closes = [100, 100.1, 100.2, 100.3, 110.0, 110.5, 110.7, 111.0]
    records = []
    for offset, close_value in enumerate(closes):
        timestamp = start + timedelta(hours=offset)
        records.append(
            make_processed_record(
                timestamp=timestamp,
                source="stocktwits",
                sentiment_label="NEUTRAL",
                sentiment_score=0.0,
            )
        )
        records.append(
            make_processed_record(
                timestamp=start + timedelta(hours=offset),
                source="yahoo_price",
                text=None,
                sentiment_label=None,
                sentiment_score=None,
                open_price=close_value - 0.1,
                high=close_value + 0.1,
                low=close_value - 0.2,
                close=close_value,
                volume=1000,
            )
        )

    feature_frame = builder.build_feature_frame(pd.DataFrame(records))

    assert feature_frame["label_volatility_spike"].max() == 1


def test_chronological_split_no_leakage() -> None:
    builder = TimeSeriesBuilder(sequence_length=3, train_ratio=0.7, val_ratio=0.15)
    frame = synthetic_feature_frame(20)

    datasets = builder.create_splits(frame)

    train_end = datasets["timestamps_train"][-1]
    val_start = datasets["timestamps_val"][0]
    test_start = datasets["timestamps_test"][0]
    assert train_end < val_start < test_start


def test_scaler_fit_on_train_only() -> None:
    builder = TimeSeriesBuilder(sequence_length=3, train_ratio=0.7, val_ratio=0.15)
    frame = synthetic_feature_frame(20)

    datasets = builder.create_splits(frame)
    scaler = datasets["scaler"]
    train_flat = datasets["X_train_raw"].reshape(-1, datasets["X_train_raw"].shape[-1])

    assert np.allclose(scaler.mean_, train_flat.mean(axis=0), atol=1e-6)


def test_sequence_shape() -> None:
    builder = TimeSeriesBuilder(sequence_length=4, train_ratio=0.7, val_ratio=0.15)
    frame = synthetic_feature_frame(20)

    datasets = builder.create_splits(frame)

    assert datasets["X_train"].shape[1] == 4
    assert datasets["X_train"].shape[2] == len(datasets["feature_columns"])


def synthetic_feature_frame(rows: int) -> pd.DataFrame:
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    data = []
    for index in range(rows):
        data.append(
            {
                "ticker": "AAPL",
                "hour_bucket": start + timedelta(hours=index),
                "sentiment_positive_count": float(index % 3),
                "sentiment_negative_count": float((index + 1) % 2),
                "sentiment_neutral_count": 1.0,
                "sentiment_score_mean": 0.1 * index,
                "sentiment_score_std": 0.01 * index,
                "total_mentions": float(index + 1),
                "reddit_mentions": 1.0,
                "twitter_mentions": 1.0,
                "news_mentions": 1.0,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000.0 + index,
                "price_return": 0.01,
                "price_return_5h": 0.02,
                "volatility_1h": 0.03,
                "volatility_6h": 0.04,
                "label_direction": [-1, 0, 1][index % 3],
                "label_return": float(index) / 100,
                "label_volatility_spike": index % 2,
            }
        )
    return pd.DataFrame(data)
