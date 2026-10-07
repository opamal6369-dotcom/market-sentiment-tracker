"""Streamlit dashboard. Run with:  streamlit run app.py"""
import streamlit as st

from src import plots
from src.pipeline import run

st.set_page_config(page_title="Market Sentiment Tracker", layout="wide")
st.title("Market Sentiment Tracker")
st.caption("Does financial news sentiment line up with stock moves? Exploratory, not investment advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL")
    days = st.slider("Lookback (days)", 30, 365, 180)
    refresh = st.checkbox("Refresh headlines (ignore cache)")
    go = st.button("Run analysis", type="primary")

if go:
    try:
        with st.spinner("Fetching data and scoring headlines (first run downloads FinBERT)..."):
            out = run(ticker, days, use_cache=not refresh)
    except Exception as e:  # show a friendly error instead of a stack trace
        st.error(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Headlines scored", len(out["scored"]))
    c2.metric("Days with sentiment", len(out["daily"]))
    c3.metric("Avg sentiment", f"{out['daily']['sentiment'].mean():+.3f}")

    st.pyplot(plots.sentiment_vs_price(out["prices"], out["daily"], out["ticker"]))
    left, right = st.columns(2)
    with left:
        st.subheader("Correlation")
        st.dataframe(out["correlations"], hide_index=True)
        st.caption("Small samples and weak correlations are expected. Check n_days and p_value.")
    with right:
        st.pyplot(plots.sentiment_vs_return(out["merged"], out["ticker"]))
    with st.expander("Scored headlines"):
        st.dataframe(out["scored"][["published", "headline", "sentiment"]], hide_index=True)
else:
    st.info("Pick a ticker in the sidebar and click **Run analysis**.")
