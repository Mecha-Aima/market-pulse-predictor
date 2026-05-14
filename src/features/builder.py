from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TimeSeriesBuilder:
    sequence_length: int = 24
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    def aggregate_sentiment(self, processed_frame: pd.DataFrame) -> pd.DataFrame:
        frame = processed_frame.copy()
        frame["date_bucket"] = (
            pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True).dt.date
        )
        frame = frame[frame["text"].notna() & frame["sentiment_label"].notna()].copy()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "date_bucket",
                    "sentiment_positive_count",
                    "sentiment_negative_count",
                    "sentiment_neutral_count",
                    "sentiment_score_mean",
                    "sentiment_score_std",
                    "total_mentions",
                    "stocktwits_mentions",
                    "news_mentions",
                ]
            )

        aggregated = (
            frame.groupby(["ticker", "date_bucket"], as_index=False)
            .apply(self._aggregate_sentiment_group, include_groups=False)
            .reset_index(drop=True)
        )
        # Drop any stray index columns produced by groupby+apply
        stray = {"level_0", "level_1", "level_2", "index"}
        aggregated = aggregated.drop(columns=[c for c in aggregated.columns if c in stray])
        return aggregated

    def build_feature_frame(self, processed_frame: pd.DataFrame) -> pd.DataFrame:
        frame = processed_frame.copy()
        ts = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True)
        frame["hour_bucket"] = ts.dt.floor("h")
        frame["date_bucket"] = ts.dt.date

        sentiment_features = self.aggregate_sentiment(frame)
        price_rows = frame[frame["source"] == "yahoo_price"].copy()
        if price_rows.empty or sentiment_features.empty:
            return pd.DataFrame()

        price_frame = (
            price_rows.groupby(["ticker", "date_bucket"], as_index=False)
            .agg(
                hour_bucket=("hour_bucket", "first"),
                open=("open", "last"),
                high=("high", "last"),
                low=("low", "last"),
                close=("close", "last"),
                volume=("volume", "last"),
            )
            .sort_values(["ticker", "date_bucket"])
        )
        price_frame = self._add_price_features(price_frame)

        # Join technicals (rsi, macd, bb_*, atr, obv) keyed by ticker+date_bucket
        tech_rows = frame[frame["source"] == "technicals"].copy()
        if not tech_rows.empty:
            tech_cols = ["ticker", "date_bucket"] + [
                c for c in tech_rows.columns
                if c in {"rsi", "macd", "macd_signal", "macd_hist",
                         "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
                         "atr", "obv"}
            ]
            tech_frame = (
                tech_rows[tech_cols]
                .groupby(["ticker", "date_bucket"], as_index=False)
                .last()  # one row per ticker/day
            )
            price_frame = price_frame.merge(tech_frame, on=["ticker", "date_bucket"], how="left")
            # Fill NaN technicals (warm-up rows) with 0
            tech_feature_cols = [c for c in tech_cols if c not in {"ticker", "date_bucket"}]
            for col in tech_feature_cols:
                if col in price_frame.columns:
                    price_frame[col] = price_frame[col].fillna(0.0)

        # Left-join: keep all price days; fill missing sentiment with 0
        merged = price_frame.merge(sentiment_features, on=["ticker", "date_bucket"], how="left")
        sentiment_cols = [
            "sentiment_positive_count", "sentiment_negative_count", "sentiment_neutral_count",
            "sentiment_score_mean", "sentiment_score_std", "total_mentions",
            "stocktwits_mentions", "news_mentions",
        ]
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        merged = merged.sort_values(["ticker", "date_bucket"]).reset_index(drop=True)
        return self._add_labels(merged)

    def create_splits(self, feature_frame: pd.DataFrame) -> dict:
        frame = feature_frame.sort_values(["ticker", "date_bucket"]).reset_index(drop=True).copy()
        frame = frame.dropna(subset=["label_direction", "label_return", "label_volatility_spike"])
        feature_columns = [
            column
            for column in frame.columns
            if column
            not in {
                "ticker",
                "date_bucket",
                "hour_bucket",
                "index",           # guard against any reset_index() leaks
                "label_direction",
                "label_return",
                "label_volatility_spike",
            }
        ]
        sequences, y_direction, y_return, y_volatility, timestamps = self._create_sequences(
            frame, feature_columns
        )
        train_end = max(1, int(len(sequences) * self.train_ratio))
        val_end = max(train_end + 1, int(len(sequences) * (self.train_ratio + self.val_ratio)))
        val_end = min(val_end, len(sequences) - 1) if len(sequences) > 2 else len(sequences)

        X_train, X_val, X_test = (
            sequences[:train_end],
            sequences[train_end:val_end],
            sequences[val_end:],
        )
        y_direction_train, y_direction_val, y_direction_test = (
            y_direction[:train_end],
            y_direction[train_end:val_end],
            y_direction[val_end:],
        )
        y_return_train, y_return_val, y_return_test = (
            y_return[:train_end],
            y_return[train_end:val_end],
            y_return[val_end:],
        )
        y_volatility_train, y_volatility_val, y_volatility_test = (
            y_volatility[:train_end],
            y_volatility[train_end:val_end],
            y_volatility[val_end:],
        )
        timestamps_train, timestamps_val, timestamps_test = (
            timestamps[:train_end],
            timestamps[train_end:val_end],
            timestamps[val_end:],
        )

        scaler = StandardScaler()
        X_train_scaled = self._fit_transform_sequences(scaler, X_train, fit=True)
        X_val_scaled = self._fit_transform_sequences(scaler, X_val, fit=False)
        X_test_scaled = self._fit_transform_sequences(scaler, X_test, fit=False)

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "X_train_raw": X_train,
            "X_val_raw": X_val,
            "X_test_raw": X_test,
            "y_direction_train": y_direction_train,
            "y_direction_val": y_direction_val,
            "y_direction_test": y_direction_test,
            "y_return_train": y_return_train,
            "y_return_val": y_return_val,
            "y_return_test": y_return_test,
            "y_volatility_train": y_volatility_train,
            "y_volatility_val": y_volatility_val,
            "y_volatility_test": y_volatility_test,
            "timestamps_train": timestamps_train,
            "timestamps_val": timestamps_val,
            "timestamps_test": timestamps_test,
            "feature_columns": feature_columns,
            "scaler": scaler,
        }

    def save_artifacts(self, datasets: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        array_names = {
            "X_train",
            "X_val",
            "X_test",
            "y_direction_train",
            "y_direction_val",
            "y_direction_test",
            "y_return_train",
            "y_return_val",
            "y_return_test",
            "y_volatility_train",
            "y_volatility_val",
            "y_volatility_test",
        }
        for name in array_names:
            np.save(output_dir / f"{name}.npy", datasets[name])
        (output_dir / "feature_columns.json").write_text(
            json.dumps(datasets["feature_columns"], indent=2)
        )
        with (output_dir / "scaler.pkl").open("wb") as handle:
            pickle.dump(datasets["scaler"], handle)

    def build_and_save(self, processed_paths: list[Path], output_dir: Path) -> dict:
        processed_frames = [pd.read_parquet(path) for path in processed_paths]
        merged = self.build_feature_frame(pd.concat(processed_frames, ignore_index=True))
        datasets = self.create_splits(merged)
        self.save_artifacts(datasets, output_dir)
        return datasets

    def _aggregate_sentiment_group(self, group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "sentiment_positive_count": int((group["sentiment_label"] == "POSITIVE").sum()),
                "sentiment_negative_count": int((group["sentiment_label"] == "NEGATIVE").sum()),
                "sentiment_neutral_count": int((group["sentiment_label"] == "NEUTRAL").sum()),
                "sentiment_score_mean": float(group["sentiment_score"].mean()),
                "sentiment_score_std": float(group["sentiment_score"].std(ddof=0) or 0.0),
                "total_mentions": int(len(group)),
                "stocktwits_mentions": int(group["source"].isin(["stocktwits"]).sum()),
                "news_mentions": int(
                    group["source"]
                    .isin(["news_rss", "yahoo_news", "alphavantage_news", "finnhub"])
                    .sum()
                ),
            }
        )

    def _add_price_features(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        grouped = price_frame.groupby("ticker", group_keys=False)
        price_frame["prev_close"] = grouped["close"].shift(1)
        price_frame["price_return"] = (
            (price_frame["close"] - price_frame["prev_close"]) / price_frame["prev_close"]
        ).fillna(0.0)
        price_frame["price_return_5h"] = (
            grouped["close"].transform(lambda values: values.pct_change(periods=5)).fillna(0.0)
        )
        price_frame["volatility_1h"] = (
            grouped["price_return"]
            .transform(lambda values: values.rolling(window=2, min_periods=1).std(ddof=0))
            .fillna(0.0)
        )
        price_frame["volatility_6h"] = (
            grouped["price_return"]
            .transform(lambda values: values.rolling(window=6, min_periods=1).std(ddof=0))
            .fillna(0.0)
        )
        return price_frame.drop(columns=["prev_close"])

    def _add_labels(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        grouped = feature_frame.groupby("ticker", group_keys=False)
        next_return = grouped["price_return"].shift(-1)
        next_volatility = grouped["volatility_1h"].shift(-1)
        rolling_mean = grouped["volatility_1h"].transform(
            lambda values: values.rolling(6, min_periods=1).mean()
        )
        rolling_std = (
            grouped["volatility_1h"]
            .transform(lambda values: values.rolling(6, min_periods=1).std(ddof=0))
            .fillna(0.0)
        )

        feature_frame["label_direction"] = np.select(
            [next_return > 0.001, next_return < -0.001],
            [1, -1],
            default=0,
        )
        feature_frame["label_return"] = next_return
        feature_frame["label_volatility_spike"] = (
            next_volatility > (rolling_mean + 2 * rolling_std)
        ).astype(int)
        return feature_frame

    def _create_sequences(
        self, frame: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sequences = []
        y_direction = []
        y_return = []
        y_volatility = []
        timestamps = []
        for _, ticker_frame in frame.groupby("ticker"):
            ticker_frame = ticker_frame.sort_values("date_bucket").reset_index(drop=True)
            for end_index in range(self.sequence_length - 1, len(ticker_frame)):
                window = ticker_frame.iloc[end_index - self.sequence_length + 1 : end_index + 1]
                sequences.append(window[feature_columns].to_numpy(dtype=float))
                target_row = ticker_frame.iloc[end_index]
                y_direction.append(int(target_row["label_direction"]))
                y_return.append(float(target_row["label_return"]))
                y_volatility.append(int(target_row["label_volatility_spike"]))
                timestamps.append(pd.Timestamp(target_row["date_bucket"]))
        return (
            np.asarray(sequences, dtype=float),
            np.asarray(y_direction, dtype=int),
            np.asarray(y_return, dtype=float),
            np.asarray(y_volatility, dtype=int),
            np.asarray(timestamps),
        )

    def _fit_transform_sequences(
        self, scaler: StandardScaler, sequences: np.ndarray, *, fit: bool
    ) -> np.ndarray:
        if sequences.size == 0:
            return sequences
        flat = sequences.reshape(-1, sequences.shape[-1])
        transformed = scaler.fit_transform(flat) if fit else scaler.transform(flat)
        return transformed.reshape(sequences.shape)
