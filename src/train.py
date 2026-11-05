"""Training entry point for both baseline and transformer models."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import hstack

from data import load_reddit_dataset, preprocess, split_dataset
from features import build_tfidf, linguistic_features
from model import TransformerConfig, build_baseline, load_transformer


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_baseline():
    print("Loading data...")
    df = preprocess(load_reddit_dataset())
    train, val, test = split_dataset(df)

    tfidf = build_tfidf(train["text"].tolist())
    X_train = hstack([tfidf.transform(train["text"]), linguistic_features(train["text"].tolist())])
    X_test  = hstack([tfidf.transform(test["text"]),  linguistic_features(test["text"].tolist())])

    model = build_baseline()
    model.fit(X_train, train["label"])

    preds = model.predict(X_test)
    report = classification_report(test["label"], preds, output_dict=True)
    print(classification_report(test["label"], preds))

    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'baseline_results.json'}")


def run_transformer(model_name: str):
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
    import evaluate

    print(f"Fine-tuning {model_name}...")
    df = preprocess(load_reddit_dataset())
    train_df, val_df, test_df = split_dataset(df)

    config = TransformerConfig(model_name=model_name)
    tokenizer, model = load_transformer(config)

    label2id = config.label2id

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length, padding="max_length")

    def encode_labels(batch):
        batch["labels"] = [label2id[l] for l in batch["label"]]
        return batch

    def make_ds(df):
        return (
            Dataset.from_pandas(df[["text", "label"]])
            .map(encode_labels, batched=True)
            .map(tokenize, batched=True)
            .remove_columns(["text", "label"])
        )

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="binary")

    args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=make_ds(train_df),
        eval_dataset=make_ds(val_df),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate(make_ds(test_df))
    print(results)

    with open(RESULTS_DIR / f"{model_name.replace('/', '_')}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--model", default="bert-base-uncased")
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline()
    else:
        run_transformer(args.model)
