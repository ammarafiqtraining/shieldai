"""
ml_model.py
===========
Layer 2 of the cascade — XGBoost classifier.

What it does:
    Trains (or loads) an XGBoost gradient-boosted tree model on labelled
    scam/ham data.  During inference it returns a probability that the
    input is a scam, based on the 28-feature vector from features.py.

Why XGBoost here?
    - Trains in minutes, not hours.
    - Naturally handles mixed numeric features.
    - Works well on CPU — no GPU needed.
    - SHAP TreeExplainer support is built-in and extremely fast.
    - Resistant to noisy/missing features.

What it does NOT do:
    Understand language semantics.  "Your account has been temporarily
    restricted for security reasons" might pass this model if it is
    phrased carefully — that is what TinyBERT in layer 3 catches.

Files produced:
    models/xgb_model.json   — serialised trained model (XGBoost native fmt)
    models/feature_names.json — ordered list of feature names for SHAP
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed.  MLModel will use fallback scoring.")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed.  Explanations will be unavailable.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR          = Path(__file__).parent / "models"
_MODEL_PATH         = _MODEL_DIR / "xgb_model.json"
_FEATURE_NAMES_PATH = _MODEL_DIR / "feature_names.json"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class MLResult:
    """
    Result from the XGBoost model.

    Attributes
    ----------
    score       : float — probability of scam (0.0–1.0)
    confidence  : str   — "high" / "medium" / "low"
    explanation : list  — top SHAP factor labels (empty if SHAP unavailable)
    skipped     : bool  — True if model not loaded (fallback mode)
    """
    score:       float
    confidence:  str          = "low"
    explanation: list[str]    = None
    skipped:     bool         = False

    def __post_init__(self):
        if self.explanation is None:
            self.explanation = []
        if self.score >= 0.75:
            self.confidence = "high"
        elif self.score >= 0.45:
            self.confidence = "medium"
        else:
            self.confidence = "low"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _build_params() -> dict:
    """
    XGBoost hyperparameters tuned for this task.

    Key choices explained:
    - max_depth=5       : shallow trees — avoids overfitting on small datasets
    - n_estimators=300  : enough trees without becoming slow
    - scale_pos_weight  : compensates for class imbalance (more ham than scam)
    - subsample=0.8     : trains each tree on 80% of data — reduces variance
    - eval_metric='aucpr': optimises area under precision-recall curve, which
                           is better than AUC-ROC for imbalanced datasets
    """
    return dict(
        max_depth          = 5,
        n_estimators       = 300,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        gamma              = 0.1,
        scale_pos_weight   = 3,        # assume ~3:1 ham:scam ratio
        use_label_encoder  = False,
        eval_metric        = "aucpr",
        random_state       = 42,
        n_jobs             = -1,       # use all CPU cores
        tree_method        = "hist",   # fast histogram-based algorithm
    )


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    eval_fraction: float = 0.15,
) -> "MLModel":
    """
    Train a new XGBoost model and save it to disk.

    Parameters
    ----------
    X              : numpy array of shape (n_samples, n_features)
    y              : numpy array of shape (n_samples,) — 0=ham, 1=scam
    feature_names  : list of feature name strings (from FeatureVector)
    eval_fraction  : fraction of data held out for early stopping

    Returns
    -------
    MLModel — loaded and ready for inference.

    How training works (plain English)
    -----------------------------------
    1.  We split the data: 85% for training, 15% for watching performance.
    2.  XGBoost builds decision trees one at a time.  Each new tree tries
        to fix the mistakes of the previous ones (this is "boosting").
    3.  Early stopping: if the model stops improving on the held-out 15%
        for 30 rounds in a row, we stop — prevents overfitting.
    4.  The result is saved so we don't have to retrain next time.
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed.  Run: pip install xgboost")

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=eval_fraction, stratify=y, random_state=42
    )

    params = _build_params()
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set          = [(X_val, y_val)],
        early_stopping_rounds = 30,
        verbose           = 50,
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(_MODEL_PATH))
    _FEATURE_NAMES_PATH.write_text(json.dumps(feature_names))

    logger.info("Model saved → %s", _MODEL_PATH)
    return MLModel(model=model, feature_names=feature_names)


def train_from_dataframe(df, text_column: str = "text", label_column: str = "label"):
    """
    Convenience wrapper: accepts a pandas DataFrame with raw text + labels,
    extracts features automatically, then trains.

    Parameters
    ----------
    df           : pandas.DataFrame
    text_column  : column name containing the raw text
    label_column : column name containing 0 (ham) or 1 (scam)

    Usage
    -----
    >>> import pandas as pd
    >>> from brain.ml_model import train_from_dataframe
    >>> df = pd.read_csv("data/sms_spam.csv")
    >>> model = train_from_dataframe(df, text_column="message", label_column="label")
    """
    from brain.features import extract, FeatureVector

    logger.info("Extracting features from %d samples …", len(df))
    vectors = [extract(str(row[text_column])) for _, row in df.iterrows()]
    feature_names = FeatureVector().feature_names

    X = np.array([v.to_list() for v in vectors], dtype=np.float32)
    y = df[label_column].astype(int).values

    logger.info("Feature matrix: %s  |  Scam ratio: %.1f%%",
                X.shape, 100 * y.mean())
    return train(X, y, feature_names)


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------
class MLModel:
    """
    Wrapper around a trained XGBoost model with SHAP explanation support.

    Instantiate via:
        model = MLModel.load()      # load from disk
        model = train(X, y, names)  # train fresh
    """

    def __init__(
        self,
        model=None,
        feature_names: Optional[list[str]] = None,
        explainer=None,
    ):
        self._model         = model
        self._feature_names = feature_names or []
        self._explainer     = explainer

    # ── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "MLModel":
        """
        Load a previously trained model from disk.

        Raises FileNotFoundError if the model has not been trained yet.
        """
        if not _XGB_AVAILABLE:
            logger.warning("xgboost unavailable — returning stub model.")
            return cls()

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {_MODEL_PATH}. "
                "Call train() or train_from_dataframe() first."
            )

        model = xgb.XGBClassifier()
        model.load_model(str(_MODEL_PATH))

        feature_names: list[str] = []
        if _FEATURE_NAMES_PATH.exists():
            feature_names = json.loads(_FEATURE_NAMES_PATH.read_text())

        explainer = None
        if _SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception as exc:
                logger.warning("Could not build SHAP explainer: %s", exc)

        logger.info("XGBoost model loaded from %s", _MODEL_PATH)
        return cls(model=model, feature_names=feature_names, explainer=explainer)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(
        self,
        feature_vector: list[float],
        top_k_explanation: int = 3,
    ) -> MLResult:
        """
        Predict scam probability for a single sample.

        Parameters
        ----------
        feature_vector      : output of FeatureVector.to_list()
        top_k_explanation   : how many SHAP factors to return

        Returns
        -------
        MLResult

        Under the hood
        --------------
        1.  Reshape the flat list into a (1, n_features) matrix.
        2.  XGBoost applies all 300 trees and averages their votes.
        3.  Sigmoid transforms the raw score → probability 0–1.
        4.  SHAP TreeExplainer computes feature contributions in ~1ms.
        5.  We return the top-k features by absolute SHAP value.
        """
        if not self.is_ready:
            return MLResult(score=0.5, skipped=True)

        arr = np.array([feature_vector], dtype=np.float32)
        proba = float(self._model.predict_proba(arr)[0][1])

        explanation: list[str] = []
        if self._explainer and self._feature_names:
            try:
                shap_vals = self._explainer.shap_values(arr)[0]
                # Sort by absolute contribution, descending
                ranked = sorted(
                    enumerate(shap_vals),
                    key=lambda t: abs(t[1]),
                    reverse=True,
                )
                explanation = [
                    self._feature_names[idx]
                    for idx, _ in ranked[:top_k_explanation]
                    if idx < len(self._feature_names)
                ]
            except Exception as exc:
                logger.debug("SHAP computation failed: %s", exc)

        return MLResult(score=round(proba, 4), explanation=explanation)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on a test set.

        Returns a dict with precision, recall, f1, and auc_pr.

        Usage
        -----
        >>> results = model.evaluate(X_test, y_test)
        >>> print(results)
        {'precision': 0.96, 'recall': 0.91, 'f1': 0.935, 'auc_pr': 0.972}
        """
        if not self.is_ready:
            return {}

        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            average_precision_score,
        )

        y_pred_proba = self._model.predict_proba(X)[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        return {
            "precision": round(precision_score(y, y_pred,  zero_division=0), 4),
            "recall":    round(recall_score(y, y_pred,     zero_division=0), 4),
            "f1":        round(f1_score(y, y_pred,         zero_division=0), 4),
            "auc_pr":    round(average_precision_score(y, y_pred_proba),     4),
        }
