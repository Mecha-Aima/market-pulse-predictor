import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingestion.base_scraper import BaseScraper


class DummyScraper(BaseScraper):
    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        return [
            {
                "id": f"{ticker}-1",
                "source": "dummy",
                "ticker": ticker,
                "timestamp": "2024-01-15T14:00:00Z",
                "text": "example",
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "score": None,
                "url": None,
            }
        ]


def test_base_scraper_save_creates_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scraper = DummyScraper()
    records = scraper.fetch("AAPL", 24)

    output_path = scraper.save(records, source="dummy", ticker="AAPL")

    assert output_path.exists()
    assert output_path.parent == tmp_path / "data" / "raw" / "dummy"
    assert json.loads(output_path.read_text()) == records


def test_yahoo_finance_fetch_returns_list(monkeypatch) -> None:
    from src.ingestion import yahoo_finance

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    history = pd.DataFrame(
        [
            {
                "Open": 182.5,
                "High": 183.2,
                "Low": 182.1,
                "Close": 183.0,
                "Volume": 45000000,
            }
        ],
        index=[pd.Timestamp(now.isoformat())],
    )
    news = [
        {
            "uuid": "news-1",
            "title": "Apple beats estimates",
            "summary": "Quarterly earnings topped expectations.",
            "link": "https://example.com/news-1",
            "providerPublishTime": int(now.timestamp()),
        }
    ]

    class FakeTicker:
        def __init__(self, ticker: str, session=None) -> None:
            self.news = news
            self.session = session

        def history(self, period: str, interval: str) -> pd.DataFrame:
            assert period == "1mo"
            assert interval == "1d"
            return history

    monkeypatch.setattr(yahoo_finance, "yf", SimpleNamespace(Ticker=FakeTicker))
    monkeypatch.setattr(yahoo_finance.YahooFinanceScraper, "_build_session", lambda self: object())

    records = yahoo_finance.YahooFinanceScraper().fetch("AAPL", 24)

    assert isinstance(records, list)
    assert len(records) == 2
    assert {
        "id",
        "source",
        "ticker",
        "timestamp",
        "text",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "score",
        "url",
    }.issubset(records[0])


def test_yahoo_rate_limit_error_triggers_retry(monkeypatch) -> None:
    from src.ingestion import yahoo_finance

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    history = pd.DataFrame(
        [{"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}],
        index=[pd.Timestamp(now.isoformat())],
    )
    calls = {"count": 0}
    rate_limit_error = type("YFRateLimitError", (Exception,), {})

    class FakeTicker:
        def __init__(self, ticker: str, session=None) -> None:
            self.news = []

        def history(self, period: str, interval: str) -> pd.DataFrame:
            calls["count"] += 1
            if calls["count"] == 1:
                raise rate_limit_error("rate limited")
            return history

    monkeypatch.setattr(
        yahoo_finance,
        "yf",
        SimpleNamespace(Ticker=FakeTicker, exceptions=SimpleNamespace(YFRateLimitError=rate_limit_error)),
    )
    monkeypatch.setattr(yahoo_finance.YahooFinanceScraper, "_build_session", lambda self: object())
    monkeypatch.setattr(yahoo_finance, "sleep", lambda _: None)

    records = yahoo_finance.YahooFinanceScraper().fetch("AAPL", 24)

    assert calls["count"] == 2
    assert len(records) == 1


def test_news_rss_deduplication(tmp_path, monkeypatch) -> None:
    from src.ingestion import news_rss

    monkeypatch.chdir(tmp_path)
    now = datetime.now(timezone.utc)
    published_parsed = now.timetuple()
    feed = SimpleNamespace(
        entries=[
            {
                "id": "reuters-1",
                "title": "AAPL lifts markets",
                "summary": "Apple moves higher.",
                "published_parsed": published_parsed,
                "link": "https://example.com/reuters-1",
            },
            {
                "id": "reuters-1",
                "title": "AAPL lifts markets",
                "summary": "Apple moves higher.",
                "published_parsed": published_parsed,
                "link": "https://example.com/reuters-1",
            },
        ]
    )

    # scraper calls httpx.get(url) first, then passes response.content to feedparser
    class FakeHttpxResponse:
        content = b"<rss/>"
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(news_rss.httpx, "get", lambda *args, **kwargs: FakeHttpxResponse())
    monkeypatch.setattr(news_rss, "feedparser", SimpleNamespace(parse=lambda _: feed))
    monkeypatch.setenv("TARGET_TICKERS", "AAPL,MSFT")

    records = news_rss.NewsRSSScraper().fetch("AAPL", 24)

    assert len(records) == 1
    assert records[0]["source"] == "news_rss"
    seen_path = tmp_path / "data" / "raw" / "news_rss" / "seen_ids.json"
    assert seen_path.exists()
    assert json.loads(seen_path.read_text()) == ["reuters-1"]



def test_stocktwits_returns_records_for_valid_ticker(tmp_path, monkeypatch) -> None:
    """StockTwits public API is defunct; scraper uses Finviz HTML news table."""
    from src.ingestion import stocktwits_scraper
    from src.ingestion.stocktwits_scraper import StockTwitsScraper

    monkeypatch.chdir(tmp_path)

    # Minimal Finviz HTML that matches _ROW_RE
    finviz_html = (
        '<table id="news-table">'
        '<tr><td align="right">May-13-24 09:30AM</td>'
        '<td><a href="https://example.com/aapl-news">AAPL surges on earnings</a></td></tr>'
        "</table>"
    )

    class FakeResponse:
        text = finviz_html
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(stocktwits_scraper.httpx, "get", lambda *args, **kwargs: FakeResponse())

    records = StockTwitsScraper().fetch("AAPL", 24)

    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["source"] == "stocktwits"
    assert records[0]["text"] == "AAPL surges on earnings"
    assert records[0]["ticker"] == "AAPL"



def test_alphavantage_returns_news_with_sentiment(monkeypatch, tmp_path) -> None:
    """AlphaVantage scraper now fetches OVERVIEW (fundamentals), not news feed."""
    from src.ingestion import alphavantage_scraper

    monkeypatch.chdir(tmp_path)

    payload = {
        "Symbol": "AAPL",
        "MarketCapitalization": "2800000000000",
        "PERatio": "28.5",
        "EPS": "6.14",
        "DividendYield": "0.005",
        "52WeekHigh": "198.23",
        "52WeekLow": "124.17",
        "50DayMovingAverage": "175.00",
        "200DayMovingAverage": "165.00",
        "Beta": "1.25",
        "ProfitMargin": "0.245",
        "RevenuePerShareTTM": "23.89",
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
    monkeypatch.setattr(alphavantage_scraper.httpx, "get", lambda *args, **kwargs: FakeResponse())

    records = alphavantage_scraper.AlphaVantageNewsScraper().fetch("AAPL", 24)

    assert len(records) == 1
    assert records[0]["source"] == "alphavantage_fundamentals"
    assert records[0]["ticker"] == "AAPL"
    assert records[0]["av_peratio"] == 28.5
    assert records[0]["av_eps"] == 6.14


@pytest.mark.integration
def test_run_all_produces_raw_files(tmp_path, monkeypatch) -> None:
    from src.ingestion import run_all

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_TICKERS", "AAPL")

    class FakeScraper(DummyScraper):
        source_name = "dummy"

    monkeypatch.setattr(run_all, "SCRAPER_CLASSES", [FakeScraper])

    result = run_all.run_all_sources(lookback_hours=24)

    assert result["dummy"] == 1
    output_dir = tmp_path / "data" / "raw" / "dummy"
    assert output_dir.exists()
    assert any(path.suffix == ".json" for path in output_dir.iterdir())
