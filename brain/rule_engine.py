"""
rule_engine.py
==============
Layer 1 of the cascade detection pipeline.

What it does:
    Scans incoming text for known scam patterns using compiled regular
    expressions.  It is intentionally simple and extremely fast (~1 ms).
    Think of it as a security guard at the front door — it catches anything
    wearing a name-tag that says "SCAM".

Why it matters:
    ~40-50 % of real-world scam traffic matches well-known templates
    (fake bank alerts, prize notifications, OTP harvesting).  Catching those
    here means the expensive ML models never have to see them.

Returns:
    A float between 0.0 (definitely clean) and 1.0 (definitely scam).
"""

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions
# Each tuple is (human_label, regex_pattern, weight)
# weight controls how much this pattern contributes to the final score.
# ---------------------------------------------------------------------------
_RAW_PATTERNS: list[tuple[str, str, float]] = [
    # --- Urgency & pressure ---
    ("urgency_act_now",      r"\b(act|respond|reply).{0,25}(now|immediately|urgent)",  0.70),
    ("urgency_expire",       r"\b(expires?|expiring|last chance).{0,30}(today|hour)",  0.65),
    ("urgency_limited",      r"\blimited.{0,15}(time|offer|slot)",                     0.55),

    # --- Account / credential harvesting ---
    ("credential_verify",    r"\b(verify|confirm|validate).{0,25}(account|identity|detail)", 0.80),
    ("credential_suspend",   r"\b(account|card|service).{0,25}(suspend|block|restrict|limit)", 0.80),
    ("credential_otp",       r"\b(otp|pin|password|passcode).{0,25}(share|send|provide|enter)", 0.90),
    ("credential_login",     r"\bclick.{0,20}(here|link|below).{0,25}(login|sign.?in|access)", 0.75),

    # --- Financial lures ---
    ("financial_prize",      r"\b(won|winner|selected|chosen|congratulations).{0,40}(prize|reward|gift|cash)", 0.85),
    ("financial_transfer",   r"\b(transfer|send|wire).{0,20}(money|fund|amount|rm|usd|\$)", 0.70),
    ("financial_refund",     r"\b(refund|rebate|cashback).{0,30}(click|claim|apply)",  0.65),
    ("financial_loan",       r"\b(easy|instant|guaranteed).{0,15}(loan|credit|approval)",    0.60),

    # --- Impersonation ---
    ("impersonate_bank",     r"\b(maybank|cimb|rhb|public bank|hong leong|ambank|bsn).{0,40}(alert|notice|verify)", 0.85),
    ("impersonate_gov",      r"\b(lhdn|pdrm|jpj|spr|kwsp|socso|epf).{0,40}(fine|penalty|verify|claim)", 0.85),
    ("impersonate_courier",  r"\b(poslaju|gdex|j&t|dhl|fedex).{0,40}(parcel|package|held|pending|fee)", 0.75),
    ("impersonate_generic",  r"\b(dear\s+customer|dear\s+user|dear\s+member)\b",        0.60),

    # --- Malicious links ---
    ("link_shortener",       r"https?://(bit\.ly|tinyurl|t\.co|ow\.ly|rb\.gy|cutt\.ly)/", 0.65),
    ("link_suspicious_tld",  r"https?://[^\s]{3,}\.(xyz|top|click|loan|work|gq|ml|cf|tk)/", 0.75),
    ("link_ip_address",      r"https?://\d{1,3}(\.\d{1,3}){3}",                         0.90),
    ("link_login_path",      r"https?://[^\s]+(login|signin|verify|secure|update)[^\s]*", 0.70),

    # --- Social engineering phrases ---
    ("social_secret",        r"\b(don.t|do not).{0,15}(tell|share|inform).{0,20}(anyone|others|family)", 0.85),
    ("social_download",      r"\b(download|install).{0,20}(app|apk|software).{0,20}(now|immediately|verify)", 0.80),
    ("social_whatsapp",      r"\b(whatsapp|telegram|signal).{0,30}(click|join|group|agent|officer)", 0.65),
]

# ---------------------------------------------------------------------------
# Compile patterns once at import time — much faster than re-compiling
# on every call.
# ---------------------------------------------------------------------------
@dataclass
class _CompiledPattern:
    label:   str
    regex:   re.Pattern
    weight:  float

def _compile_patterns() -> list[_CompiledPattern]:
    compiled = []
    for label, pattern, weight in _RAW_PATTERNS:
        try:
            compiled.append(_CompiledPattern(
                label  = label,
                regex  = re.compile(pattern, re.IGNORECASE | re.UNICODE),
                weight = weight,
            ))
        except re.error as exc:
            logger.error("Failed to compile pattern '%s': %s", label, exc)
    return compiled

_PATTERNS: list[_CompiledPattern] = _compile_patterns()

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    """
    The result returned by the rule engine for a single piece of text.

    Attributes
    ----------
    score       : float  — 0.0 (clean) to 1.0 (scam).
    matched     : list   — human-readable labels of every pattern that fired.
    top_match   : str    — the single highest-weight match, or 'none'.
    is_definite : bool   — True when score > 0.88 (rule engine is certain).
    """
    score:       float
    matched:     list[str]  = field(default_factory=list)
    top_match:   str        = "none"
    is_definite: bool       = False


def _normalize(text: str) -> str:
    """
    Normalise Unicode so that lookalike characters collapse to their ASCII
    equivalents before pattern matching.

    Example:  "pаypal" (Cyrillic 'а') → "paypal"
    """
    return unicodedata.normalize("NFKC", text)


def score(text: str) -> RuleResult:
    """
    Score a piece of text against all compiled scam patterns.

    Parameters
    ----------
    text : str
        Raw input — SMS body, email body, chat message, etc.

    Returns
    -------
    RuleResult
        See dataclass definition above.

    How the score is calculated
    ---------------------------
    Each matching pattern contributes its weight.  We use a *noisy-OR*
    combination so that multiple weak signals add up, but a single strong
    signal alone can push the score high:

        combined = 1 - ∏(1 - wᵢ)   for all matching patterns i

    This keeps the score in [0, 1] and naturally handles overlap.
    """
    if not text or not isinstance(text, str):
        return RuleResult(score=0.0)

    clean_text = _normalize(text)

    matched_patterns: list[_CompiledPattern] = []
    for p in _PATTERNS:
        if p.regex.search(clean_text):
            matched_patterns.append(p)

    if not matched_patterns:
        return RuleResult(score=0.0)

    # Noisy-OR combination
    combined = 1.0
    for p in matched_patterns:
        combined *= (1.0 - p.weight)
    final_score = round(1.0 - combined, 4)

    top = max(matched_patterns, key=lambda p: p.weight)

    return RuleResult(
        score       = min(final_score, 1.0),
        matched     = [p.label for p in matched_patterns],
        top_match   = top.label,
        is_definite = final_score > 0.88,
    )
