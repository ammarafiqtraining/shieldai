"""
pipeline.py
===========
The Brain — master orchestrator of the three-layer cascade.

What it does:
    Takes any raw text input and returns a final risk score (0–100) with
    a human-readable explanation.  It coordinates:

        Layer 1  →  Rule Engine   (rule_engine.py)
        Layer 2  →  XGBoost ML    (ml_model.py)
        Layer 3  →  TinyBERT NLP  (nlp_model.py)

The cascade principle (think of it as a funnel):
    ┌─────────────────────────────────────────────────────┐
    │  Input text                                          │
    │      │                                               │
    │      ▼                                               │
    │  [Rule Engine] ── obvious scam / obvious clean ──►  │
    │      │ (ambiguous)                                   │
    │      ▼                                               │
    │  [XGBoost ML]  ── high/low confidence ──────────►   │
    │      │ (still ambiguous)                             │
    │      ▼                                               │
    │  [TinyBERT NLP] ── final semantic verdict ──────►   │
    │                                                      │
    │  [Ensemble Scorer] → Risk Score 0-100 + Explanation │
    └─────────────────────────────────────────────────────┘

    ~60% of traffic exits at Layer 1 (fast, cheap).
    ~25% exits at Layer 2 (medium speed).
    ~15% reaches Layer 3 (expensive but thorough).

Output:
    AnalysisResult dataclass — score, verdict, explanation, timing, channel.
"""

import logging
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Optional

from brain.rule_engine import score as rule_score, RuleResult
from brain.features    import extract as extract_features
from brain.ml_model    import MLModel, MLResult
from brain.nlp_model   import NLPModel, NLPResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cascade thresholds
# Tune these on the validation set using the PR curve.
# ---------------------------------------------------------------------------
_RULE_EXIT_HIGH   = 0.88   # Rule engine confident → SCAM (skip ML + NLP)
_RULE_EXIT_LOW    = 0.08   # Rule engine confident → CLEAN (skip ML + NLP)
_ML_EXIT_HIGH     = 0.82   # XGBoost confident     → SCAM (skip NLP)
_ML_EXIT_LOW      = 0.15   # XGBoost confident     → CLEAN (skip NLP)

# Ensemble weights — must sum to 1.0
_W_RULE = 0.20
_W_ML   = 0.35
_W_NLP  = 0.45

# Risk score thresholds (0–100 scale)
_THRESH_CLEAN     = 30
_THRESH_SUSPICIOUS = 60
_THRESH_HIGH      = 80


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """
    Final output of the Brain pipeline for a single input.

    Attributes
    ----------
    analysis_id   : str   — unique UUID for this analysis (for audit logs)
    score         : int   — risk score 0–100
    verdict       : str   — "clean" / "suspicious" / "high_risk" / "scam"
    verdict_color : str   — "green" / "orange" / "red" / "dark_red"
    explanation   : str   — human-readable reason (for end users)
    factors       : list  — machine-readable contributing factor labels
    channel       : str   — "message" / "email" / "url"
    layers_used   : list  — which layers were invoked ["rule", "ml", "nlp"]
    latency_ms    : float — total processing time in milliseconds
    rule_score    : float — raw rule engine score (0.0–1.0)
    ml_score      : float — raw XGBoost score    (0.0–1.0), None if skipped
    nlp_score     : float — raw TinyBERT score   (0.0–1.0), None if skipped
    """
    analysis_id:   str
    score:         int
    verdict:       str
    verdict_color: str
    explanation:   str
    factors:       list[str]        = field(default_factory=list)
    channel:       str              = "message"
    layers_used:   list[str]        = field(default_factory=list)
    latency_ms:    float            = 0.0
    rule_score:    float            = 0.0
    ml_score:      Optional[float]  = None
    nlp_score:     Optional[float]  = None

    def to_dict(self) -> dict:
        return {
            "analysis_id":   self.analysis_id,
            "score":         self.score,
            "verdict":       self.verdict,
            "verdict_color": self.verdict_color,
            "explanation":   self.explanation,
            "factors":       self.factors,
            "channel":       self.channel,
            "layers_used":   self.layers_used,
            "latency_ms":    round(self.latency_ms, 2),
            "raw_scores": {
                "rule": round(self.rule_score, 4),
                "ml":   round(self.ml_score,  4) if self.ml_score  is not None else None,
                "nlp":  round(self.nlp_score, 4) if self.nlp_score is not None else None,
            },
        }


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------
def _verdict_from_score(score: int) -> tuple[str, str]:
    """Map numeric score → (verdict_label, colour)."""
    if score <= _THRESH_CLEAN:
        return "clean",      "green"
    if score <= _THRESH_SUSPICIOUS:
        return "suspicious", "orange"
    if score <= _THRESH_HIGH:
        return "high_risk",  "red"
    return "scam",           "dark_red"


_EXPLANATION_TEMPLATES = {
    "clean":      "No significant scam indicators detected. This appears to be legitimate.",
    "suspicious": "Some patterns suggest caution. Review before clicking links or sharing information.",
    "high_risk":  "Multiple scam indicators found. Do NOT click links or provide personal information.",
    "scam":       "HIGH CONFIDENCE SCAM detected. Block and report this message immediately.",
}

_FACTOR_LABELS = {
    # Rule engine
    "urgency_act_now":       "Urgency pressure language",
    "urgency_expire":        "Expiry/deadline pressure",
    "credential_verify":     "Account verification request",
    "credential_suspend":    "Account suspension threat",
    "credential_otp":        "OTP/password harvesting attempt",
    "financial_prize":       "Fake prize/reward lure",
    "financial_transfer":    "Suspicious money transfer request",
    "impersonate_bank":      "Bank impersonation detected",
    "impersonate_gov":       "Government agency impersonation",
    "link_ip_address":       "Direct IP address URL (never legitimate)",
    "link_shortener":        "Suspicious URL shortener",
    "social_secret":         "Social engineering — secrecy request",
    # XGBoost features
    "urgency_word_count":    "High urgency word density",
    "has_ip_url":            "IP-based URL present",
    "has_suspicious_tld":    "Suspicious domain extension",
    "caps_ratio":            "Excessive capital letters",
    "brand_mention_count":   "Brand name mentioned (possible impersonation)",
    "financial_word_count":  "High financial vocabulary density",
    "has_short_url":         "URL shortener detected",
    "max_domain_entropy":    "High-entropy (random-looking) domain",
}


def _humanise_factors(factors: list[str]) -> list[str]:
    """Convert machine labels → readable strings."""
    return [_FACTOR_LABELS.get(f, f.replace("_", " ").title()) for f in factors]


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------
class Brain:
    """
    The central intelligence of the fraud detection system.

    Usage
    -----
    >>> brain = Brain.load()
    >>> result = brain.analyze("Your account has been suspended. Verify now: http://1.2.3.4/login")
    >>> print(result.score, result.verdict, result.explanation)
    92 scam HIGH CONFIDENCE SCAM detected ...

    Loading
    -------
    Brain.load()         — loads all models from disk (production use)
    Brain.load_stub()    — loads only rule engine (useful before training)
    """

    def __init__(
        self,
        ml_model:  Optional[MLModel]  = None,
        nlp_model: Optional[NLPModel] = None,
    ):
        self._ml  = ml_model  or MLModel()
        self._nlp = nlp_model or NLPModel()

    # ── Constructors ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "Brain":
        """Load all three layers.  Safe to call even if models not yet trained."""
        logger.info("Initialising Brain — loading all models …")
        ml  = _safe_load(MLModel.load,  "XGBoost")
        nlp = _safe_load(NLPModel.load, "TinyBERT")
        return cls(ml_model=ml, nlp_model=nlp)

    @classmethod
    def load_stub(cls) -> "Brain":
        """Load only the rule engine (no ML/NLP). Useful for quick tests."""
        return cls()

    # ── Core analysis ─────────────────────────────────────────────────────────
    def analyze(
        self,
        text:    str,
        channel: str = "message",
    ) -> AnalysisResult:
        """
        Analyze a piece of text and return a risk assessment.

        Parameters
        ----------
        text    : str — raw input (SMS body, email body, URL, chat message)
        channel : str — "message" | "email" | "url"

        Returns
        -------
        AnalysisResult

        Cascade walkthrough (step by step)
        ------------------------------------
        Step 1 — Sanitise:
            Normalise Unicode, strip leading/trailing whitespace.
            This prevents obfuscation tricks like Cyrillic lookalikes.

        Step 2 — Rule Engine:
            Run compiled regex patterns.  If score > 0.88 → definite scam.
            If score < 0.08 → definite clean.  Either way, we're done fast.

        Step 3 — XGBoost (if ambiguous):
            Extract 28 numerical features, feed to XGBoost.
            High confidence exits here; only ambiguous cases continue.

        Step 4 — TinyBERT (if still ambiguous):
            Full semantic understanding.  Catches social engineering that
            avoids keywords.

        Step 5 — Ensemble:
            Weighted combination of all active layer scores → 0–100 risk score.
            SHAP factors from XGBoost + matched rule labels → explanation.

        Step 6 — Return:
            AnalysisResult with score, verdict, explanation, timing, audit ID.
        """
        t_start = time.perf_counter()
        aid     = str(uuid.uuid4())
        text    = _sanitise(text)
        layers  = []
        factors = []

        if not text:
            return _empty_result(aid, channel)

        # ── Layer 1: Rule Engine ──────────────────────────────────────────────
        r_result: RuleResult = rule_score(text)
        layers.append("rule")
        factors.extend(r_result.matched[:3])

        if r_result.score >= _RULE_EXIT_HIGH:
            final_score = int(min(r_result.score * 105, 100))   # slight boost for definite
            return _build_result(
                aid, final_score, factors, channel, layers, t_start,
                rule=r_result.score, ml=None, nlp=None,
            )

        if r_result.score <= _RULE_EXIT_LOW:
            return _build_result(
                aid, int(r_result.score * 100), factors, channel, layers, t_start,
                rule=r_result.score, ml=None, nlp=None,
            )

        # ── Layer 2: XGBoost ─────────────────────────────────────────────────
        fv          = extract_features(text)
        ml_result: MLResult = self._ml.predict(fv.to_list())
        layers.append("ml")
        factors.extend(ml_result.explanation[:3])

        if not ml_result.skipped:
            if ml_result.score >= _ML_EXIT_HIGH:
                combo = _weighted(r_result.score, ml_result.score, None)
                return _build_result(
                    aid, _to_100(combo), factors, channel, layers, t_start,
                    rule=r_result.score, ml=ml_result.score, nlp=None,
                )
            if ml_result.score <= _ML_EXIT_LOW:
                combo = _weighted(r_result.score, ml_result.score, None)
                return _build_result(
                    aid, _to_100(combo), factors, channel, layers, t_start,
                    rule=r_result.score, ml=ml_result.score, nlp=None,
                )

        # ── Layer 3: TinyBERT ─────────────────────────────────────────────────
        nlp_result: NLPResult = self._nlp.predict(text[:512])
        layers.append("nlp")

        combo = _weighted(
            r_result.score,
            ml_result.score if not ml_result.skipped else 0.5,
            nlp_result.score if not nlp_result.skipped else 0.5,
        )
        return _build_result(
            aid, _to_100(combo), factors, channel, layers, t_start,
            rule=r_result.score,
            ml=ml_result.score   if not ml_result.skipped  else None,
            nlp=nlp_result.score if not nlp_result.skipped else None,
        )

    def analyze_email(
        self,
        subject:  str = "",
        body:     str = "",
        sender:   str = "",
        reply_to: str = "",
    ) -> AnalysisResult:
        """
        Analyze an email by combining all available fields into a single text.

        Subject and sender mismatch is a strong impersonation signal —
        concatenating them lets the models see both in context.
        """
        combined = f"From: {sender}\nReply-To: {reply_to}\nSubject: {subject}\n\n{body}"
        result   = self.analyze(combined, channel="email")
        return result

    def analyze_url(self, url: str) -> AnalysisResult:
        """
        Analyze a single URL string.

        Passes the URL as text so the rule engine can match URL patterns,
        and the feature extractor pulls URL-specific signals.
        """
        return self.analyze(url, channel="url")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.strip()[:10_000]       # hard cap — prevents DoS via huge input


def _weighted(r: float, m: float, n: Optional[float]) -> float:
    """
    Compute weighted ensemble score.
    If NLP is not available, redistribute its weight to ML.
    """
    if n is None:
        w_r = _W_RULE / (_W_RULE + _W_ML)
        w_m = _W_ML   / (_W_RULE + _W_ML)
        return w_r * r + w_m * m
    return _W_RULE * r + _W_ML * m + _W_NLP * n


def _to_100(score: float) -> int:
    return int(min(max(round(score * 100), 0), 100))


def _build_result(
    aid:     str,
    score:   int,
    factors: list[str],
    channel: str,
    layers:  list[str],
    t_start: float,
    rule:    float,
    ml:      Optional[float],
    nlp:     Optional[float],
) -> AnalysisResult:
    verdict, colour = _verdict_from_score(score)
    readable_factors = _humanise_factors(list(dict.fromkeys(factors)))  # dedup, preserve order
    base_explanation = _EXPLANATION_TEMPLATES[verdict]
    if readable_factors:
        top3 = "; ".join(readable_factors[:3])
        explanation = f"{base_explanation} Top signals: {top3}."
    else:
        explanation = base_explanation

    return AnalysisResult(
        analysis_id   = aid,
        score         = score,
        verdict       = verdict,
        verdict_color = colour,
        explanation   = explanation,
        factors       = readable_factors,
        channel       = channel,
        layers_used   = layers,
        latency_ms    = (time.perf_counter() - t_start) * 1000,
        rule_score    = rule,
        ml_score      = ml,
        nlp_score     = nlp,
    )


def _empty_result(aid: str, channel: str) -> AnalysisResult:
    return AnalysisResult(
        analysis_id   = aid,
        score         = 0,
        verdict       = "clean",
        verdict_color = "green",
        explanation   = "Empty input — nothing to analyze.",
        channel       = channel,
    )


def _safe_load(loader_fn, name: str):
    try:
        return loader_fn()
    except FileNotFoundError:
        logger.warning("%s model not found — will be skipped in cascade.", name)
        return None
    except Exception as exc:
        logger.error("Failed to load %s model: %s", name, exc)
        return None
