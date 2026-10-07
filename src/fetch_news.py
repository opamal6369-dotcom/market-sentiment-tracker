"""Collect headlines for a ticker.

Sources:
  * yfinance  - free, no key, but only returns RECENT headlines.
  * finnhub   - free API key, can return about a year of history.
Set FINNHUB_API_KEY in your environment to use Finnhub.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

COLUMNS = ["published", "headline", "source"]


def _to_utc(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True)
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_yfinance_news(ticker: str) -> pd.DataFrame:
    rows = []
    for item in yf.Ticker(ticker).news or []:
        content = item.get("content", item)  # yfinance changed its schema over time
        title = content.get("title")
        published = _to_utc(content.get("pubDate") or content.get("providerPublishTime"))
        if title and published is not None and not pd.isna(published):
            rows.append({"published": published, "headline": title, "source": "yfinance"})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_finnhub_news(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": ticker, "from": start, "to": end, "token": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    rows = [
        {
            "published": datetime.fromtimestamp(item["datetime"], tz=timezone.utc),
            "headline": item["headline"],
            "source": "finnhub",
        }
        for item in resp.json()
        if item.get("headline") and item.get("datetime")
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_news(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Use Finnhub if a key is set, otherwise fall back to yfinance."""
    key = os.getenv("FINNHUB_API_KEY")
    df = fetch_finnhub_news(ticker, start, end, key) if key else fetch_yfinance_news(ticker)
    df = df.drop_duplicates(subset="headline").copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.sort_values("published").reset_index(drop=True)
