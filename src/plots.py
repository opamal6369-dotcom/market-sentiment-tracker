"""Charts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def sentiment_vs_price(prices: pd.DataFrame, daily: pd.DataFrame, ticker: str):
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(prices["date"], prices["close"], color="tab:blue", label="Close")
    ax1.set_ylabel("Close price")
    ax2 = ax1.twinx()
    colors = ["tab:green" if s >= 0 else "tab:red" for s in daily["sentiment"]]
    ax2.bar(daily["date"], daily["sentiment"], color=colors, alpha=0.5, width=0.8)
    ax2.set_ylabel("Avg headline sentiment (-1 to +1)")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax1.set_title(f"{ticker}: daily headline sentiment vs price")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def sentiment_vs_return(merged: pd.DataFrame, ticker: str, target: str = "next_return"):
    sub = merged[["sentiment", target]].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sub["sentiment"], sub[target] * 100, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Avg headline sentiment")
    ax.set_ylabel(f"{target.replace('_', ' ')} (%)")
    ax.set_title(f"{ticker}: sentiment vs {target.replace('_', ' ')}")
    fig.tight_layout()
    return fig
