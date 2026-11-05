"""Linguistic feature engineering (baseline models)."""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


FIRST_PERSON = re.compile(r"\b(i|me|my|myself|mine)\b")
NEGATION = re.compile(r"\b(no|not|never|nothing|nobody|nowhere|nor|neither|cannot|can't|won't|don't|didn't|doesn't|isn't|wasn't|aren't|weren't)\b")
NEGATIVE_EMOTION = re.compile(r"\b(sad|depressed|hopeless|worthless|empty|numb|lonely|miserable|anxious|afraid|exhausted|tired)\b")


def linguistic_features(texts: list[str]) -> np.ndarray:
    """
    Hand-crafted LIWC-inspired features:
      0  text length (log)
      1  first-person pronoun rate
      2  negation rate
      3  negative-emotion word rate
      4  avg word length
    """
    rows = []
    for text in texts:
        words = text.split()
        n = max(len(words), 1)
        rows.append([
            np.log1p(len(text)),
            len(FIRST_PERSON.findall(text)) / n,
            len(NEGATION.findall(text)) / n,
            len(NEGATIVE_EMOTION.findall(text)) / n,
            np.mean([len(w) for w in words]) if words else 0,
        ])
    return np.array(rows, dtype=np.float32)


def build_tfidf(train_texts: list[str], **kwargs) -> TfidfVectorizer:
    defaults = dict(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    vec = TfidfVectorizer(**{**defaults, **kwargs})
    vec.fit(train_texts)
    return vec
