"""Data loading and preprocessing utilities."""

import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_twitter_dataset(split: str = "train") -> pd.DataFrame:
    """Load the sampled Twitter depression dataset from Hugging Face."""
    ds = load_dataset("indominousx/depression-detection", split=split)
    return ds.to_pandas()


def clean_text(text: str) -> str:
    """Basic social media text cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                   # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)              # strip hashtag symbol
    text = re.sub(r"[^\w\s']", " ", text)              # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.len() > 10]
    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) splits stratified on label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
