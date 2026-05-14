"""
Local technical indicator computation from Yahoo Finance OHLCV data.

Computes RSI, MACD, Bollinger Bands, ATR, and OBV using pandas only —
zero API calls, no AV premium tier needed.

Run standalone:
    python -m src.ingestion.technicals

Or call from your pipeline:
    from src.ingestion.technicals import TechnicalsEnricher
    TechnicalsEnricher().run()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

LOGGER = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TECHNICALS_DIR = RAW_DIR / "technicals"


class TechnicalsEnricher:
    """Reads Yahoo OHLCV JSON files and writes per-ticker technical indicator records."""

    source_name = "technicals"

    def run(self) -> dict[str, int]:
        if pd is None:
            raise RuntimeError("pandas is required for local technical indicators")

        TECHNICALS_DIR.mkdir(parents=True, exist_ok=True)
        yahoo_dir = RAW_DIR / "yahoo"
        if not yahoo_dir.exists():
            LOGGER.warning("No Yahoo data found at %s — run Yahoo scraper first", yahoo_dir)
            return {}

        # Group files by ticker
        ticker_files: dict[str, list[Path]] = {}
        for f in sorted(yahoo_dir.glob("*.json")):
            parts = f.stem.split("_")
            ticker = parts[0]
            ticker_files.setdefault(ticker, []).append(f)

        counts: dict[str, int] = {}
        for ticker, files in ticker_files.items():
            records = self._process_ticker(ticker, files)
            if records:
                self._save(ticker, records)
                counts[ticker] = len(records)
                print(f"  ✓ TechnicalsEnricher [{ticker}]: {len(records)} indicator records")
            else:
                print(f"  ○ TechnicalsEnricher [{ticker}]: not enough OHLCV data")
        return counts

    def _process_ticker(self, ticker: str, files: list[Path]) -> list[dict]:
        # Load all price records from all files for this ticker
        price_rows = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                rows = data if isinstance(data, list) else [data]
                price_rows.extend(
                    r
                    for r in rows
                    if r.get("source") == "yahoo_price" and r.get("close") is not None
                )
            except Exception as e:
                LOGGER.debug("Skipping %s: %s", f, e)

        if len(price_rows) < 14:
            return []  # Need at least 14 candles for RSI

        df = pd.DataFrame(price_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        df = self._add_rsi(df, period=14)
        df = self._add_macd(df)
        df = self._add_bollinger(df, period=20)
        df = self._add_atr(df, period=14)
        df = self._add_obv(df)

        records = []
        for ts, row in df.iterrows():
            record: dict[str, Any] = {
                "id": f"{ticker}-tech-{ts.isoformat()}",
                "source": "technicals",
                "ticker": ticker,
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "text": None,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else None,
                "score": None,
                "url": None,
            }
            for col in df.columns:
                if col not in ("open", "high", "low", "close", "volume"):
                    val = row.get(col)
                    record[col] = float(val) if pd.notna(val) else None
            records.append(record)

        return records

    # ── Indicator implementations ──────────────────────────────────────────

    @staticmethod
    def _add_rsi(df: "pd.DataFrame", period: int = 14) -> "pd.DataFrame":
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_macd(df: "pd.DataFrame") -> "pd.DataFrame":
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def _add_bollinger(df: "pd.DataFrame", period: int = 20) -> "pd.DataFrame":
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + 2 * std
        df["bb_mid"] = sma
        df["bb_lower"] = sma - 2 * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    @staticmethod
    def _add_atr(df: "pd.DataFrame", period: int = 14) -> "pd.DataFrame":
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
        return df

    @staticmethod
    def _add_obv(df: "pd.DataFrame") -> "pd.DataFrame":
        direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df["obv"] = (direction * df["volume"]).cumsum()
        return df

    def _save(self, ticker: str, records: list[dict]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = TECHNICALS_DIR / f"{ticker}_{ts}.json"
        out_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    enricher = TechnicalsEnricher()
    counts = enricher.run()
    total = sum(counts.values())
    print(f"\n✓ Technicals computed for {len(counts)} ticker(s): {total} total records")
