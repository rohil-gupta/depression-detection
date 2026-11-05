"""Model definitions: baseline (sklearn) and transformer fine-tune."""

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── Baseline ──────────────────────────────────────────────────────────────────

def build_baseline(C: float = 1.0) -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("scale", MaxAbsScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced")),
    ])


# ── Transformer ───────────────────────────────────────────────────────────────

MENTAL_BERT = "mental/mental-bert-base-uncased"
DEFAULT_MODEL = "bert-base-uncased"


@dataclass
class TransformerConfig:
    model_name: str = DEFAULT_MODEL
    num_labels: int = 2
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    label2id: dict = field(default_factory=lambda: {"non-depression": 0, "depression": 1})
    id2label: dict = field(default_factory=lambda: {0: "non-depression", 1: "depression"})


def load_transformer(config: TransformerConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
    )
    return tokenizer, model
