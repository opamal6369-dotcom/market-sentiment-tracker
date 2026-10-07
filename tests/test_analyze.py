import numpy as np
import pandas as pd

from src.analyze import aggregate_daily, assign_trading_day, correlations, merge_sentiment_prices
from src.fetch_prices import add_returns

# Trading days Jan 2-10 2025 (Jan 4-5 is a weekend)
DAYS = pd.to_datetime(
    ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]
)


def ts(s):
    return pd.Series(pd.to_datetime([s], utc=True))


def test_intraday_headline_maps_to_same_day():
    # 10:00 ET = 15:00 UTC (EST)
    assert assign_trading_day(ts("2025-01-07 15:00"), DAYS).iloc[0] == pd.Timestamp("2025-01-07")


def test_after_close_headline_maps_to_next_day():
    # 17:00 ET = 22:00 UTC
    assert assign_trading_day(ts("2025-01-07 22:00"), DAYS).iloc[0] == pd.Timestamp("2025-01-08")


def test_friday_after_close_maps_to_monday():
    assert assign_trading_day(ts("2025-01-03 22:00"), DAYS).iloc[0] == pd.Timestamp("2025-01-06")


def test_weekend_headline_maps_to_monday():
    assert assign_trading_day(ts("2025-01-04 15:00"), DAYS).iloc[0] == pd.Timestamp("2025-01-06")


def test_after_last_trading_day_is_dropped():
    assert pd.isna(assign_trading_day(ts("2025-01-10 22:00"), DAYS).iloc[0])


def test_aggregate_daily_means_and_counts():
    scored = pd.DataFrame(
        {
            "published": pd.to_datetime(["2025-01-07 15:00", "2025-01-07 16:00", "2025-01-08 15:00"], utc=True),
            "sentiment": [0.5, -0.1, 0.2],
        }
    )
    daily = aggregate_daily(scored, DAYS)
    row = daily[daily["date"] == "2025-01-07"].iloc[0]
    assert np.isclose(row["sentiment"], 0.2)
    assert row["n_headlines"] == 2


def test_add_returns_next_day_is_shifted():
    df = add_returns(pd.DataFrame({"date": DAYS[:3], "close": [100.0, 110.0, 99.0]}))
    assert np.isclose(df.loc[1, "return"], 0.10)
    assert np.isclose(df.loc[0, "next_return"], 0.10)
    assert np.isnan(df.loc[2, "next_return"])


def test_correlation_perfect_relationship():
    sent = np.linspace(-1, 1, 20)
    merged = pd.DataFrame({"sentiment": sent, "return": sent * 0.01, "next_return": sent * 0.02})
    assert np.allclose(correlations(merged)["pearson_r"], 1.0)


def test_correlation_returns_nan_when_sample_too_small():
    merged = pd.DataFrame({"sentiment": [0.1, 0.2], "return": [0.01, 0.02], "next_return": [0.0, 0.01]})
    assert correlations(merged)["pearson_r"].isna().all()


def test_merge_keeps_only_days_with_sentiment():
    prices = add_returns(pd.DataFrame({"date": DAYS, "close": np.arange(100, 107, dtype=float)}))
    daily = pd.DataFrame({"date": [DAYS[1], DAYS[3]], "sentiment": [0.3, -0.2], "n_headlines": [2, 1]})
    assert len(merge_sentiment_prices(prices, daily)) == 2
