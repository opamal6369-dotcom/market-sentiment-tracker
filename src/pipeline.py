"""End-to-end pipeline + command-line entry point.

Example:
    python -m src.pipeline AAPL --days 180
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import analyze, fetch_news, fetch_prices, plots, sentiment

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def run(ticker: str, days: int = 180, use_cache: bool = True) -> dict:
    ticker = ticker.upper().strip()
    end = date.today()
    start = end - timedelta(days=days)
    DATA_DIR.mkdir(exist_ok=True)

    prices = fetch_prices.fetch_prices(ticker, start.isoformat(), end.isoformat())

    scored_path = DATA_DIR / f"{ticker}_scored.csv"
    if use_cache and scored_path.exists():
        scored = pd.read_csv(scored_path)
        scored["published"] = pd.to_datetime(scored["published"], utc=True)
    else:
        news = fetch_news.fetch_news(ticker, start.isoformat(), end.isoformat())
        if news.empty:
            raise ValueError(f"No headlines found for {ticker}.")
        scores = sentiment.score_headlines(news["headline"].tolist())
        scored = pd.concat([news.reset_index(drop=True), scores], axis=1)
        scored.to_csv(scored_path, index=False)

    daily = analyze.aggregate_daily(scored, prices["date"])
    merged = analyze.merge_sentiment_prices(prices, daily)
    corr = analyze.correlations(merged)
    return {
        "ticker": ticker,
        "prices": prices,
        "scored": scored,
        "daily": daily,
        "merged": merged,
        "correlations": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headline sentiment vs stock returns")
    parser.add_argument("ticker")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--refresh", action="store_true", help="ignore cached headlines")
    args = parser.parse_args()

    out = run(args.ticker, args.days, use_cache=not args.refresh)
    print(f"\nHeadlines scored: {len(out['scored'])}")
    print(f"Trading days with sentiment: {len(out['daily'])}")
    print(out["correlations"].to_string(index=False))

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    plots.sentiment_vs_price(out["prices"], out["daily"], out["ticker"]).savefig(
        fig_dir / f"{out['ticker']}_timeline.png", dpi=150
    )
    plots.sentiment_vs_return(out["merged"], out["ticker"]).savefig(
        fig_dir / f"{out['ticker']}_scatter.png", dpi=150
    )
    print(f"Saved charts to {fig_dir}/")


if __name__ == "__main__":
    main()
