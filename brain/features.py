"""
features.py
===========
Converts raw text / URL / email input into a flat numerical feature vector
that the XGBoost model can consume.

What it does:
    Extracts ~35 hand-crafted signals that are strong predictors of scam
    content.  These are things a trained analyst would look at manually —
    we just automate and quantify them.

Why hand-crafted features?
    Neural networks learn features automatically, but XGBoost needs numbers.
    Hand-crafted features also make the model *explainable* — when SHAP
    highlights "high urgency word count" as the top factor, a non-technical
    user immediately understands what that means.

Feature groups:
    1. Lexical   — character-level statistics (length, digit ratio, etc.)
    2. Urgency   — count of pressure/urgency vocabulary
    3. Financial — count of money-related vocabulary
    4. Link      — URL-specific signals
    5. Structure — formatting red-flags (ALL CAPS ratio, punctuation abuse)
    6. Identity  — impersonation signals
"""

import re
import math
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Optional

# ---------------------------------------------------------------------------
# Vocabulary lists — kept small and high-signal deliberately
# ---------------------------------------------------------------------------
_URGENCY_WORDS = frozenset([
    "urgent", "immediately", "expire", "expires", "expiring",
    "deadline", "limited", "hurry", "asap", "final", "last",
    "alert", "warning", "suspended", "blocked", "restricted",
])

_FINANCIAL_WORDS = frozenset([
    "money", "cash", "transfer", "fund", "payment", "invoice",
    "refund", "reward", "prize", "bonus", "loan", "credit",
    "bank", "account", "wallet", "bitcoin", "crypto",
])

_IDENTITY_BRANDS = frozenset([
    "maybank", "cimb", "rhb", "publicbank", "hlb", "ambank", "bsn",
    "paypal", "amazon", "apple", "google", "microsoft", "netflix",
    "lhdn", "kwsp", "epf", "socso", "pdrm", "jpj",
    "poslaju", "gdex", "dhl", "fedex",
])

_GREETING_PATTERNS = [
    re.compile(r"\bdear\s+(customer|user|member|valued|sir|madam)\b", re.I),
    re.compile(r"\b(hello|hi)\s+(there|friend|dear)\b", re.I),
]

_URL_PATTERN = re.compile(
    r"https?://[^\s\)<>\"\']+"
    r"|www\.[a-z0-9\-]+\.[a-z]{2,}[^\s\)<>\"\']*",
    re.I,
)

_SUSPICIOUS_TLDS = frozenset([
    ".xyz", ".top", ".click", ".loan", ".work",
    ".gq", ".ml", ".cf", ".tk", ".pw",
])

_IP_URL = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}")

_SHORT_DOMAINS = frozenset([
    "bit.ly", "tinyurl.com", "t.co", "ow.ly",
    "rb.gy", "cutt.ly", "is.gd", "v.gd",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _entropy(s: str) -> float:
    """
    Shannon entropy of a string.
    High entropy in a domain name is a red-flag (randomly generated domains).
    """
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    total = len(s)
    return -sum((v / total) * math.log2(v / total) for v in freq.values())


def _extract_urls(text: str) -> list[str]:
    return _URL_PATTERN.findall(text)


def _url_features(urls: list[str]) -> dict[str, float]:
    """Derive URL-specific numeric features from a list of URLs found in text."""
    if not urls:
        return {
            "url_count": 0,
            "has_ip_url": 0,
            "has_short_url": 0,
            "has_suspicious_tld": 0,
            "max_url_length": 0,
            "max_domain_entropy": 0.0,
            "has_login_path": 0,
            "has_http_only": 0,
        }

    ip_url           = 0
    short_url        = 0
    suspicious_tld   = 0
    login_path       = 0
    http_only        = 0
    max_len          = 0
    max_entropy      = 0.0

    for url in urls:
        try:
            parsed = urlparse(url if "://" in url else "https://" + url)
            domain = parsed.netloc.lower().lstrip("www.")
            path   = (parsed.path or "").lower()

            if _IP_URL.match(url):
                ip_url = 1
            if any(domain == d for d in _SHORT_DOMAINS):
                short_url = 1
            if any(url.lower().endswith(tld) or f"{tld}/" in url.lower()
                   for tld in _SUSPICIOUS_TLDS):
                suspicious_tld = 1
            if re.search(r"(login|signin|verify|secure|update|confirm)", path):
                login_path = 1
            if url.startswith("http://"):
                http_only = 1

            ent = _entropy(domain)
            max_entropy = max(max_entropy, round(ent, 4))
            max_len     = max(max_len, len(url))
        except Exception:
            continue

    return {
        "url_count":            len(urls),
        "has_ip_url":           ip_url,
        "has_short_url":        short_url,
        "has_suspicious_tld":   suspicious_tld,
        "max_url_length":       max_len,
        "max_domain_entropy":   max_entropy,
        "has_login_path":       login_path,
        "has_http_only":        http_only,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
@dataclass
class FeatureVector:
    """
    A flat, named dictionary of numeric features ready for XGBoost.

    All values are either:
    - Integers  (0 / 1 for flags, raw counts for frequencies)
    - Floats    (ratios, entropy, normalised scores)

    Attributes are also accessible as a plain dict via .to_dict().
    """
    # Lexical
    char_count:             int   = 0
    word_count:             int   = 0
    digit_ratio:            float = 0.0
    special_char_ratio:     float = 0.0
    caps_ratio:             float = 0.0
    avg_word_length:        float = 0.0
    text_entropy:           float = 0.0

    # Urgency
    urgency_word_count:     int   = 0
    urgency_ratio:          float = 0.0
    exclamation_count:      int   = 0

    # Financial
    financial_word_count:   int   = 0
    has_currency_symbol:    int   = 0
    has_amount_pattern:     int   = 0

    # Identity / impersonation
    brand_mention_count:    int   = 0
    has_generic_greeting:   int   = 0

    # Link features (8 fields from _url_features)
    url_count:              int   = 0
    has_ip_url:             int   = 0
    has_short_url:          int   = 0
    has_suspicious_tld:     int   = 0
    max_url_length:         int   = 0
    max_domain_entropy:     float = 0.0
    has_login_path:         int   = 0
    has_http_only:          int   = 0

    # Structure
    question_mark_count:    int   = 0
    line_count:             int   = 0
    has_phone_number:       int   = 0
    has_email_address:      int   = 0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    def to_list(self) -> list[float]:
        return list(self.to_dict().values())

    @property
    def feature_names(self) -> list[str]:
        return list(self.__dict__.keys())


def extract(text: str) -> FeatureVector:
    """
    Extract all features from a raw text string.

    Parameters
    ----------
    text : str
        Any raw input (SMS, email body, chat message).

    Returns
    -------
    FeatureVector
        Named numeric features.

    Step-by-step walkthrough
    ------------------------
    1.  Normalise Unicode so Cyrillic lookalikes don't slip through.
    2.  Compute character-level statistics (length, ratios).
    3.  Tokenize into words, count urgency/financial vocabulary.
    4.  Find all URLs in the text and extract URL-level features.
    5.  Check structural red-flags (excessive punctuation, phone numbers).
    6.  Return everything packed into FeatureVector.
    """
    if not text or not isinstance(text, str):
        return FeatureVector()

    text  = _normalize(text)
    lower = text.lower()
    chars = len(text)

    if chars == 0:
        return FeatureVector()

    # ── Lexical ──────────────────────────────────────────────────────────────
    digits        = sum(c.isdigit() for c in text)
    special       = sum(not c.isalnum() and not c.isspace() for c in text)
    caps          = sum(c.isupper() for c in text)
    words         = re.findall(r"\b\w+\b", lower)
    word_count    = len(words)
    avg_wl        = round(sum(len(w) for w in words) / max(word_count, 1), 3)

    # ── Urgency ───────────────────────────────────────────────────────────────
    urgency_hits  = sum(1 for w in words if w in _URGENCY_WORDS)

    # ── Financial ────────────────────────────────────────────────────────────
    fin_hits      = sum(1 for w in words if w in _FINANCIAL_WORDS)
    currency_flag = int(bool(re.search(r"[\$£€¥₹rm]\s*\d", lower)))
    amount_flag   = int(bool(re.search(r"\d[\d,\.]+\s*(rm|usd|myr|dollar)", lower)))

    # ── Identity ─────────────────────────────────────────────────────────────
    brand_hits    = sum(1 for brand in _IDENTITY_BRANDS if brand in lower)
    greeting_flag = int(any(p.search(lower) for p in _GREETING_PATTERNS))

    # ── URLs ─────────────────────────────────────────────────────────────────
    urls          = _extract_urls(text)
    url_feats     = _url_features(urls)

    # ── Structure ────────────────────────────────────────────────────────────
    phone_flag    = int(bool(re.search(
        r"(\+?60|0)[1-9]\d{7,9}|\b\d{3}[\s\-]\d{3,4}[\s\-]\d{4}\b", text
    )))
    email_flag    = int(bool(re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", lower)))

    return FeatureVector(
        # Lexical
        char_count             = chars,
        word_count             = word_count,
        digit_ratio            = round(digits / chars, 4),
        special_char_ratio     = round(special / chars, 4),
        caps_ratio             = round(caps / chars, 4),
        avg_word_length        = avg_wl,
        text_entropy           = round(_entropy(lower), 4),

        # Urgency
        urgency_word_count     = urgency_hits,
        urgency_ratio          = round(urgency_hits / max(word_count, 1), 4),
        exclamation_count      = text.count("!"),

        # Financial
        financial_word_count   = fin_hits,
        has_currency_symbol    = currency_flag,
        has_amount_pattern     = amount_flag,

        # Identity
        brand_mention_count    = brand_hits,
        has_generic_greeting   = greeting_flag,

        # Link
        **url_feats,

        # Structure
        question_mark_count    = text.count("?"),
        line_count             = text.count("\n") + 1,
        has_phone_number       = phone_flag,
        has_email_address      = email_flag,
    )
