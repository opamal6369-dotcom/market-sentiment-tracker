"""Score headlines with FinBERT (a BERT model fine-tuned on financial text)."""
from __future__ import annotations

import pandas as pd

MODEL_NAME = "ProsusAI/finbert"
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline  # imported lazily: heavy import

        _classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    return _classifier


def score_headlines(headlines, batch_size: int = 16) -> pd.DataFrame:
    """Return columns: positive, negative, neutral, sentiment.

    `sentiment` = P(positive) - P(negative), ranging from -1 to +1.
    """
    if not headlines:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "sentiment"])
    results = _get_classifier()(list(headlines), truncation=True, batch_size=batch_size)
    rows = []
    for probs in results:
        p = {r["label"].lower(): r["score"] for r in probs}
        rows.append(
            {
                "positive": p.get("positive", 0.0),
                "negative": p.get("negative", 0.0),
                "neutral": p.get("neutral", 0.0),
                "sentiment": p.get("positive", 0.0) - p.get("negative", 0.0),
            }
        )
    return pd.DataFrame(rows)
