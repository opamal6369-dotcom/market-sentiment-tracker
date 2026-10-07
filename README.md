# Market Sentiment Tracker

Does the tone of financial news line up with how a stock moves? This project pulls recent
headlines for a ticker, scores each one with **FinBERT** (a BERT model fine-tuned on financial
text), aggregates sentiment per trading day, and measures its correlation with **same-day** and
**next-day** returns.

> Exploratory analysis only. Not investment advice.

## How it works

1. **Prices** – daily closes and returns via `yfinance`.
2. **Headlines** – `yfinance` news (recent only) or the Finnhub company-news API (about a year of
   history, free key).
3. **Sentiment** – FinBERT outputs P(positive), P(negative), P(neutral). Score = P(pos) - P(neg).
4. **Alignment** – headlines published after 4pm ET, or on weekends/holidays, are mapped to the
   *next* trading day so sentiment is never paired with a price move that already happened.
5. **Analysis** – Pearson correlation (with p-value and sample size) of daily sentiment vs
   same-day and next-day returns.
6. **Dashboard** – Streamlit app with timeline, scatter plot, and the scored headlines.

## Setup

```bash
git clone https://github.com/<your-username>/market-sentiment-tracker.git
cd market-sentiment-tracker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for longer news history: get a free key at https://finnhub.io and run
`export FINNHUB_API_KEY=your_key` (see `.env.example`).

## Usage

```bash
# Command line: prints correlations and saves charts to figures/
python -m src.pipeline AAPL --days 180

# Dashboard
streamlit run app.py

# Tests
pytest
```

The first run downloads the FinBERT model (~440 MB). Scored headlines are cached in `data/`.

## Project structure

```
src/
  fetch_prices.py   prices + returns
  fetch_news.py     headlines (yfinance / Finnhub)
  sentiment.py      FinBERT scoring
  analyze.py        trading-day alignment, aggregation, correlation
  plots.py          matplotlib charts
  pipeline.py       end-to-end + CLI
app.py              Streamlit dashboard
tests/              pytest unit tests for the alignment/aggregation/correlation logic
```

## Results

<!-- TODO: fill this in AFTER you run it, using your real output. Example structure:
Tickers: ...  | Window: ... | Headlines scored: ... | Days with sentiment: ...
Same-day correlation: r = ..., p = ... (n = ...)
Next-day correlation: r = ..., p = ... (n = ...)
Include a screenshot of the dashboard: ![dashboard](figures/dashboard.png)
-->

## Limitations

- Free news APIs only give a limited history, so samples are small and correlations noisy.
- Headline sentiment is a crude signal; a model can misread sarcasm, context, or old news.
- Correlation is not causation, and a result on one ticker/window may not generalize.
- No transaction costs, no trading strategy, no out-of-sample test.

## Ideas for next steps

- Compare across multiple tickers and sectors.
- Add a Reddit source and compare against news.
- Out-of-sample test or simple baseline (e.g., previous-day return) to see if sentiment adds anything.

## License

MIT
