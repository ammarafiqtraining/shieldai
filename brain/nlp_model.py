"""
nlp_model.py
============
Layer 3 of the cascade — TinyBERT fine-tuned for scam detection.

What it does:
    Uses a small pre-trained transformer model (TinyBERT / DistilBERT) to
    understand the *meaning* of text, not just keywords or statistics.

    This catches sophisticated scams that deliberately avoid trigger words
    — e.g. a Business Email Compromise message that sounds completely
    professional but is requesting a fraudulent wire transfer.

Why TinyBERT over full BERT?
    - 4.4M parameters vs 110M (BERT-base) → 25x smaller
    - Runs on CPU in ~150–300ms per message
    - Fine-tuned on scam data it matches or exceeds full BERT on this task
    - Fits in RAM on a basic server (< 100MB)

Fine-tuning:
    Call fine_tune() with your labelled DataFrame.
    If no fine-tuned model exists, the raw pre-trained model is used as
    a fallback (accuracy will be lower but the system won't crash).

Files produced:
    models/nlp_finetuned/   — Hugging Face model + tokenizer checkpoint
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch/transformers not installed.  NLPModel will be skipped.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MODEL_DIR    = Path(__file__).parent / "models" / "nlp_finetuned"
_BASE_MODEL   = "prajjwal1/bert-tiny"          # 4.4M params — fast on CPU
_FALLBACK_MODEL = "distilbert-base-uncased"    # 66M params — if tiny unavailable
_MAX_LENGTH   = 128                            # token limit per message
_LABEL2ID     = {"ham": 0, "scam": 1}
_ID2LABEL     = {0: "ham", 1: "scam"}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class NLPResult:
    """
    Result from the NLP model.

    Attributes
    ----------
    score       : float — probability of scam (0.0–1.0)
    label       : str   — "scam" or "ham"
    confidence  : float — same as score, kept for readability
    skipped     : bool  — True if model not available (fallback mode)
    """
    score:      float
    label:      str   = "ham"
    confidence: float = 0.0
    skipped:    bool  = False

    def __post_init__(self):
        self.label      = "scam" if self.score >= 0.5 else "ham"
        self.confidence = self.score if self.label == "scam" else 1.0 - self.score


# ---------------------------------------------------------------------------
# Dataset wrapper for Hugging Face Trainer
# ---------------------------------------------------------------------------
class _ScamDataset:
    """
    Minimal PyTorch Dataset that wraps tokenized encodings + labels.
    Used internally by fine_tune() — not part of the public API.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class NLPModel:
    """
    Wrapper around a fine-tuned TinyBERT classifier.

    Instantiate via:
        model = NLPModel.load()                 # load fine-tuned from disk
        model = NLPModel.load_pretrained()      # load base (not fine-tuned)
        model = fine_tune(df, ...)              # train and return
    """

    def __init__(self, tokenizer=None, model=None, device: str = "cpu"):
        self._tokenizer = tokenizer
        self._model     = model
        self._device    = device

    @property
    def is_ready(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    # ── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "NLPModel":
        """
        Load the fine-tuned model from disk.
        Falls back to pretrained base model if fine-tuning hasn't been done.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — NLPModel will be skipped.")
            return cls()

        if _MODEL_DIR.exists():
            logger.info("Loading fine-tuned NLP model from %s", _MODEL_DIR)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
                model     = AutoModelForSequenceClassification.from_pretrained(
                    str(_MODEL_DIR)
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                return cls(tokenizer=tokenizer, model=model, device=device)
            except Exception as exc:
                logger.error("Failed to load fine-tuned model: %s", exc)

        logger.warning("No fine-tuned model found.  Using pretrained base.")
        return cls.load_pretrained()

    @classmethod
    def load_pretrained(cls) -> "NLPModel":
        """
        Load the raw pre-trained TinyBERT without fine-tuning.
        Useful for testing the pipeline before training data is ready.
        """
        if not _TORCH_AVAILABLE:
            return cls()

        for base in [_BASE_MODEL, _FALLBACK_MODEL]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(base)
                model     = AutoModelForSequenceClassification.from_pretrained(
                    base, num_labels=2,
                    id2label=_ID2LABEL, label2id=_LABEL2ID,
                    ignore_mismatched_sizes=True,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                logger.info("Loaded pretrained model: %s", base)
                return cls(tokenizer=tokenizer, model=model, device=device)
            except Exception as exc:
                logger.warning("Could not load %s: %s", base, exc)

        return cls()

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> NLPResult:
        """
        Predict scam probability for a single text input.

        Parameters
        ----------
        text : str — raw message text (max 512 chars recommended)

        Returns
        -------
        NLPResult

        Under the hood
        --------------
        1.  Tokenizer splits text into subword tokens (WordPiece).
        2.  Tokens are padded/truncated to _MAX_LENGTH (128).
        3.  TinyBERT processes the token sequence through 2 transformer layers.
        4.  The [CLS] token representation is fed into a 2-class linear head.
        5.  Softmax converts raw logits → probability [ham_prob, scam_prob].
        6.  We return scam_prob as the score.
        """
        if not self.is_ready:
            return NLPResult(score=0.5, skipped=True)

        try:
            inputs = self._tokenizer(
                text,
                return_tensors  = "pt",
                truncation      = True,
                max_length      = _MAX_LENGTH,
                padding         = "max_length",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1)
                scam_prob = float(probs[0][1])

            return NLPResult(score=round(scam_prob, 4))

        except Exception as exc:
            logger.error("NLP inference failed: %s", exc)
            return NLPResult(score=0.5, skipped=True)

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[NLPResult]:
        """
        Predict scam probability for a list of texts efficiently.

        Batching is ~4-8x faster than calling predict() in a loop because
        the transformer processes multiple inputs simultaneously.

        Parameters
        ----------
        texts      : list of raw text strings
        batch_size : number of texts per forward pass (tune to available RAM)
        """
        if not self.is_ready:
            return [NLPResult(score=0.5, skipped=True) for _ in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            try:
                inputs = self._tokenizer(
                    chunk,
                    return_tensors  = "pt",
                    truncation      = True,
                    max_length      = _MAX_LENGTH,
                    padding         = True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs  = torch.softmax(logits, dim=-1)
                for prob_row in probs:
                    results.append(NLPResult(score=round(float(prob_row[1]), 4)))
            except Exception as exc:
                logger.error("Batch NLP failed at chunk %d: %s", i, exc)
                results.extend([NLPResult(score=0.5, skipped=True)] * len(chunk))

        return results


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def fine_tune(
    df,
    text_column:  str   = "text",
    label_column: str   = "label",
    epochs:       int   = 3,
    batch_size:   int   = 16,
    learning_rate:float = 2e-5,
    eval_fraction:float = 0.15,
) -> NLPModel:
    """
    Fine-tune TinyBERT on a labelled scam/ham dataset.

    Parameters
    ----------
    df            : pandas.DataFrame
    text_column   : column with raw message text
    label_column  : column with integer labels (0=ham, 1=scam)
    epochs        : number of full passes over the training data
                    (3 is usually enough; 5 if accuracy is low)
    batch_size    : messages per gradient update (lower if OOM)
    learning_rate : how fast the model learns (2e-5 is a safe default)
    eval_fraction : fraction held out for validation

    Returns
    -------
    NLPModel — loaded fine-tuned model.

    What fine-tuning does (plain English)
    ---------------------------------------
    TinyBERT was pre-trained on Wikipedia & BookCorpus — it understands
    English grammar but knows nothing about scams.  Fine-tuning takes the
    pre-trained weights and nudges them using your scam/ham examples.
    After 3 epochs the model has learned scam-specific patterns while
    retaining its language understanding.  This is dramatically cheaper
    than training from scratch.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch/transformers not installed.")

    from sklearn.model_selection import train_test_split

    texts  = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size  = eval_fraction,
        stratify   = labels,
        random_state = 42,
    )

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL, num_labels=2,
        id2label=_ID2LABEL, label2id=_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def _tokenize(texts_):
        return tokenizer(
            texts_, truncation=True, max_length=_MAX_LENGTH, padding="max_length"
        )

    import torch as _torch
    train_enc = _tokenize(X_train)
    val_enc   = _tokenize(X_val)

    train_dataset = _ScamDataset(
        {k: _torch.tensor(v) for k, v in train_enc.items()},
        _torch.tensor(y_train),
    )
    val_dataset   = _ScamDataset(
        {k: _torch.tensor(v) for k, v in val_enc.items()},
        _torch.tensor(y_val),
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir              = str(_MODEL_DIR),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate           = learning_rate,
        weight_decay            = 0.01,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        logging_steps           = 50,
        no_cuda                 = not _torch.cuda.is_available(),
        report_to               = "none",            # no wandb/tensorboard
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
    )

    logger.info("Starting fine-tuning on %d samples …", len(X_train))
    trainer.train()
    trainer.save_model(str(_MODEL_DIR))
    tokenizer.save_pretrained(str(_MODEL_DIR))
    logger.info("Fine-tuned model saved → %s", _MODEL_DIR)

    return NLPModel.load()
