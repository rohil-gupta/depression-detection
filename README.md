# Depression Detection from Social Media Text

An NLP project that detects signs of depression in social media posts using a progression of models — from classical ML baselines to fine-tuned transformers.

> **Disclaimer:** This is a research project only. The model is not a clinical diagnostic tool and should never be used as a substitute for professional mental health evaluation.

---

## Dataset

We use the [Twitter Depression Sampled dataset](https://huggingface.co/datasets/indominousx/depression-detection) (Hugging Face), which contains posts from r/depression and control subreddits with binary labels.

Download is handled automatically by `src/data.py` via the `datasets` library.

---

## Project Structure

```
depression-detection/
├── data/
│   ├── raw/           # original data (gitignored)
│   └── processed/     # cleaned splits (gitignored)
├── notebooks/
│   └── 01_eda.ipynb   # exploratory data analysis
├── src/
│   ├── data.py        # loading & preprocessing
│   ├── features.py    # linguistic feature engineering
│   ├── model.py       # model definitions
│   └── train.py       # training entry point
├── results/           # JSON metric outputs
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**Baseline (TF-IDF + Logistic Regression):**
```bash
python src/train.py --mode baseline
```

**Fine-tune BERT:**
```bash
python src/train.py --mode transformer --model bert-base-uncased
```

**Fine-tune Mental-BERT** (pre-trained on mental health corpora):
```bash
python src/train.py --mode transformer --model mental/mental-bert-base-uncased
```

**Exploratory analysis:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Results

| Model | F1 (depression) | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | — | — |
| BERT fine-tuned | — | — |
| Mental-BERT fine-tuned | — | — |

*Fill in after running experiments.*

---

## Methods

### Baseline
- Text cleaning (lowercasing, URL/mention removal)
- TF-IDF (unigrams + bigrams, 20k features)
- Hand-crafted LIWC-inspired features: first-person pronoun rate, negation rate, negative-emotion word rate
- Logistic Regression with class balancing

### Transformer
- Fine-tune `bert-base-uncased` or `mental/mental-bert-base-uncased`
- Max sequence length 128, AdamW lr=2e-5, 3 epochs
- Early stopping on validation F1

---

## Ethical Considerations

- Posts are publicly available on Twitter and used under research norms.
- The binary label ("depression" / "non-depression") is a proxy for community membership, not a clinical diagnosis.
- Model outputs carry uncertainty and should not inform clinical or personal decisions.
- If you or someone you know is struggling, please contact the **988 Suicide & Crisis Lifeline** (call or text 988 in the US).
