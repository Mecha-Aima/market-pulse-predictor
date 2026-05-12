from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    nltk = None
    SentimentIntensityAnalyzer = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SentimentBackend(Protocol):
    def score(self, text: str) -> float: ...


@dataclass
class VaderBackend:
    analyzer: object

    def score(self, text: str) -> float:
        return float(self.analyzer.polarity_scores(text)["compound"])


class KeywordBackend:
    POSITIVE = {"rally", "strong", "beat", "bullish", "climbing", "growth", "gain"}
    NEGATIVE = {"crash", "fear", "recession", "loom", "drop", "bearish", "loss"}

    def score(self, text: str) -> float:
        lowered = text.lower()
        positive = sum(word in lowered for word in self.POSITIVE)
        negative = sum(word in lowered for word in self.NEGATIVE)
        if positive == negative:
            return 0.0
        total = positive + negative
        return (positive - negative) / total


class FinBERTBackend:
    def __init__(self) -> None:
        from transformers import pipeline

        self.pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch and torch.cuda.is_available() else -1,
        )

    def score(self, text: str) -> float:
        result = self.pipeline(text, truncation=True)[0]
        score = float(result["score"])
        if result["label"].lower() == "negative":
            return -score
        if result["label"].lower() == "neutral":
            return 0.0
        return score


class SentimentAnalyzer:
    def __init__(self) -> None:
        self.backend = self._build_backend()

    def analyze(self, text: str) -> dict:
        score = self.backend.score(text)
        label = self._label_from_score(score)
        return {"label": label, "score": score}

    def batch_analyze(self, records: list[dict]) -> list[dict]:
        enriched = []
        for record in records:
            text = record.get("text")
            av_result = self._alphavantage_sentiment(record)
            if av_result is not None:
                enriched.append(
                    {
                        **record,
                        "sentiment_label": av_result["label"],
                        "sentiment_score": av_result["score"],
                    }
                )
                continue
            if text is None:
                enriched.append({**record, "sentiment_label": None, "sentiment_score": None})
                continue
            result = self.analyze(text)
            enriched.append(
                {
                    **record,
                    "sentiment_label": result["label"],
                    "sentiment_score": result["score"],
                }
            )
        return enriched

    def _build_backend(self) -> SentimentBackend:
        use_finbert = os.getenv("USE_FINBERT") == "1"
        if use_finbert and torch and torch.cuda.is_available():
            try:
                return FinBERTBackend()
            except Exception:  # pragma: no cover
                pass
        if SentimentIntensityAnalyzer is not None:
            try:
                return VaderBackend(self._load_vader())
            except Exception:  # pragma: no cover
                pass
        return KeywordBackend()

    def _load_vader(self):
        if nltk is not None:
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:  # pragma: no cover
                try:
                    nltk.download("vader_lexicon", quiet=True)
                except Exception:
                    pass
        return SentimentIntensityAnalyzer()

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= 0.05:
            return "POSITIVE"
        if score <= -0.05:
            return "NEGATIVE"
        return "NEUTRAL"

    @staticmethod
    def _alphavantage_sentiment(record: dict) -> dict | None:
        raw_label = record.get("av_sentiment_label")
        if raw_label is None:
            return None

        mapped_label = {
            "Bullish": "POSITIVE",
            "Bearish": "NEGATIVE",
            "Neutral": "NEUTRAL",
        }.get(raw_label)
        if mapped_label is None:
            return None

        score = record.get("av_sentiment_score")
        return {
            "label": mapped_label,
            "score": float(score) if score is not None else 0.0,
        }
