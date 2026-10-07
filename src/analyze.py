"""Align headlines with trading days, aggregate, and measure correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16


def assign_trading_day(published: pd.Series, trading_days) -> pd.Series:
    """Map each timestamp to the trading day whose price it could influence.

    Headlines published at/after 4pm ET (or on weekends/holidays) are mapped
    to the next trading day, so sentiment never gets paired with a price
    move that already happened.
    """
    ts = pd.to_datetime(published, utc=True).dt.tz_convert(MARKET_TZ)
    day = ts.dt.tz_localize(None).dt.normalize()
    day = day + pd.to_timedelta((ts.dt.hour >= MARKET_CLOSE_HOUR).astype(int), unit="D")
    days = pd.DatetimeIndex(trading_days).normalize().sort_values()
    pos = days.searchsorted(day.values, side="left")
    valid = pos < len(days)
    out = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns]")
    out.loc[valid] = days[pos[valid]].values
    return out


def aggregate_daily(scored: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Average sentiment per trading day. Expects columns: published, sentiment."""
    df = scored.copy()
    df["date"] = assign_trading_day(df["published"], trading_days)
    df = df.dropna(subset=["date"])
    return (
        df.groupby("date")
        .agg(sentiment=("sentiment", "mean"), n_headlines=("sentiment", "size"))
        .reset_index()
    )


def merge_sentiment_prices(prices: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    return prices.merge(daily, on="date", how="inner")


def correlations(merged: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Pearson correlation of sentiment vs same-day and next-day returns."""
    rows = []
    for col, label in [("return", "same-day return"), ("next_return", "next-day return")]:
        sub = merged[["sentiment", col]].dropna()
        n = len(sub)
        if n < min_n or sub["sentiment"].nunique() < 2 or sub[col].nunique() < 2:
            rows.append({"target": label, "n_days": n, "pearson_r": np.nan, "p_value": np.nan})
            continue
        r, p = stats.pearsonr(sub["sentiment"], sub[col])
        rows.append({"target": label, "n_days": n, "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows)
