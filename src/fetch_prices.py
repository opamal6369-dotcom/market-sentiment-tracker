"""Download daily price history and compute returns."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame with columns: date, close, return, next_return.

    `return` is the close-to-close return on that day.
    `next_return` is the following trading day's return, so you can test
    whether today's sentiment predicts tomorrow's move.
    """
    raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker!r} ({start} to {end}).")
    df = raw[["Close"]].rename(columns={"Close": "close"}).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.reset_index()
    return add_returns(df)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["return"] = df["close"].pct_change()
    df["next_return"] = df["return"].shift(-1)
    return df
