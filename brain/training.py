"""
training.py
===========
One-stop training script.

Run this ONCE after downloading your datasets.  It:
    1. Loads and merges SMS + email datasets
    2. Cleans and balances the data
    3. Trains XGBoost (Layer 2)
    4. Fine-tunes TinyBERT (Layer 3)
    5. Evaluates both models and prints a report

Usage
-----
    # From the project root:
    python -m brain.training

    # Or from Python:
    from brain.training import train_all
    train_all(data_dir="data/")

Expected data directory layout
-------------------------------
    data/
      sms_spam.csv        — columns: text, label  (0=ham, 1=spam)
      enron_email.csv     — columns: text, label  (0=ham, 1=spam)
      phishtank.csv       — columns: url, label   (0=legit, 1=phish)

All CSV files must have a header row.
Labels must be integer 0 (clean) or 1 (scam).

Output
------
    brain/models/xgb_model.json
    brain/models/feature_names.json
    brain/models/nlp_finetuned/        ← fine-tuned TinyBERT
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_csv(path: Path, text_col: str, label_col: str) -> "pd.DataFrame":
    """Load a single CSV and normalise column names."""
    df = pd.read_csv(path, usecols=[text_col, label_col])
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    logger.info("  Loaded %s — %d rows (scam: %d)", path.name, len(df), df["label"].sum())
    return df


def _load_all_datasets(data_dir: Path) -> "pd.DataFrame":
    """
    Discover and load every supported dataset file in data_dir.

    Supported filenames and expected columns:
        sms_spam.csv        text, label
        enron_email.csv     text, label
        phishtank.csv       url, label   (url column renamed to text)
        spamassassin.csv    text, label
    """
    file_map = {
        "sms_spam.csv":      ("text", "label"),
        "enron_email.csv":   ("text", "label"),
        "phishtank.csv":     ("url",  "label"),
        "spamassassin.csv":  ("text", "label"),
    }

    frames = []
    for filename, (text_col, label_col) in file_map.items():
        path = data_dir / filename
        if path.exists():
            try:
                frames.append(_load_csv(path, text_col, label_col))
            except Exception as exc:
                logger.warning("Could not load %s: %s", filename, exc)
        else:
            logger.info("  %s not found — skipping.", filename)

    if not frames:
        raise FileNotFoundError(
            f"No dataset files found in {data_dir}.  "
            "Expected at least one of: sms_spam.csv, enron_email.csv, phishtank.csv"
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total samples: %d  |  Scam: %d  |  Ham: %d",
                len(combined), combined["label"].sum(), (combined["label"] == 0).sum())
    return combined


def _clean_dataset(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Apply data quality steps:
    1. Remove exact-text duplicates.
    2. Remove rows with text shorter than 5 characters.
    3. Cap text at 2000 characters (rare extreme outliers waste compute).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= 5]
    df["text"] = df["text"].str[:2000]
    logger.info("Cleaned: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_all(
    data_dir:       str  = "data",
    skip_nlp:       bool = False,
    nlp_epochs:     int  = 3,
    nlp_batch_size: int  = 16,
) -> None:
    """
    Full training pipeline — call this once after data is ready.

    Parameters
    ----------
    data_dir       : path to directory containing CSV dataset files
    skip_nlp       : set True to skip TinyBERT fine-tuning (saves ~30 min)
    nlp_epochs     : fine-tuning epochs (3 is usually sufficient)
    nlp_batch_size : batch size for fine-tuning (lower if RAM is limited)

    What happens step by step
    --------------------------
    1. Load all CSVs from data_dir into a unified DataFrame.
    2. Clean: dedup, length filter, text cap.
    3. Extract feature vectors → train XGBoost → evaluate on test split.
    4. (Optional) Fine-tune TinyBERT → save checkpoint.
    5. Print final metrics report.
    """
    if not _PANDAS or not _NUMPY:
        raise ImportError("pandas and numpy are required.  pip install pandas numpy")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

    logger.info("=" * 60)
    logger.info("BRAIN TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1 & 2: Load + clean ──────────────────────────────────────────────
    logger.info("\n[1/4] Loading datasets from %s …", data_path)
    df = _load_all_datasets(data_path)
    df = _clean_dataset(df)

    # ── Step 3: XGBoost ───────────────────────────────────────────────────────
    logger.info("\n[2/4] Training XGBoost model …")
    from brain.ml_model import train_from_dataframe
    from sklearn.model_selection import train_test_split
    import numpy as np

    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )

    xgb_model = train_from_dataframe(train_df)

    logger.info("\n[3/4] Evaluating XGBoost on test set …")
    from brain.features import extract, FeatureVector
    X_test = np.array(
        [extract(t).to_list() for t in test_df["text"]],
        dtype=np.float32,
    )
    y_test = test_df["label"].values
    metrics = xgb_model.evaluate(X_test, y_test)
    _print_metrics("XGBoost", metrics)

    # ── Step 4: TinyBERT ──────────────────────────────────────────────────────
    if not skip_nlp:
        logger.info("\n[4/4] Fine-tuning TinyBERT …")
        try:
            from brain.nlp_model import fine_tune
            fine_tune(
                train_df,
                text_column   = "text",
                label_column  = "label",
                epochs        = nlp_epochs,
                batch_size    = nlp_batch_size,
            )
            logger.info("TinyBERT fine-tuning complete.")
        except ImportError:
            logger.warning("torch/transformers not installed — skipping NLP fine-tuning.")
        except Exception as exc:
            logger.error("NLP fine-tuning failed: %s", exc)
    else:
        logger.info("\n[4/4] Skipping NLP fine-tuning (skip_nlp=True).")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE  — models saved to brain/models/")
    logger.info("Run the API with:  uvicorn api.main:app --reload")
    logger.info("=" * 60)


def _print_metrics(name: str, metrics: dict) -> None:
    logger.info("  ── %s Results ──", name)
    for k, v in metrics.items():
        bar = "✓" if (k == "f1" and v >= 0.90) or \
                     (k == "precision" and v >= 0.93) or \
                     (k == "recall" and v >= 0.88) else "·"
        logger.info("    %s  %-12s %.4f", bar, k, v)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Brain models.")
    parser.add_argument("--data-dir",   default="data",  help="Path to dataset directory")
    parser.add_argument("--skip-nlp",   action="store_true", help="Skip TinyBERT fine-tuning")
    parser.add_argument("--nlp-epochs", type=int, default=3,  help="Fine-tuning epochs")
    args = parser.parse_args()

    train_all(
        data_dir   = args.data_dir,
        skip_nlp   = args.skip_nlp,
        nlp_epochs = args.nlp_epochs,
    )
