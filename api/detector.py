"""
api/detector.py
===============
Singleton wrapper around the Brain pipeline.

Responsibilities:
  1. Load and cache the Brain instance once at startup.
  2. Run the detection cascade and enrich with campaign/persona/VT data.
  3. Convert the AnalysisResult to the full API response schema.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from brain.pipeline import Brain, AnalysisResult
from brain.rule_engine import score as _run_rule_engine

from api.models import (
    AnalyzeResponse, MatchedPattern, RawScores, VTResult, VTEngine,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern label mapping: brain key → (UI display label, weight)
# Labels MUST match what RULES/TTPMAP/WHY_DESC in shieldai_v3.html expect.
# ---------------------------------------------------------------------------
_PATTERN_LABELS: dict[str, tuple[str, float]] = {
    "urgency_act_now":      ("Urgency pressure",       0.70),
    "urgency_expire":       ("Expiry/deadline threat",  0.65),
    "urgency_limited":      ("Scarcity pressure",       0.55),
    "credential_verify":    ("Verify account link",     0.80),
    "credential_suspend":   ("Account suspension",      0.82),
    "credential_otp":       ("OTP harvesting",          0.90),
    "credential_login":     ("Fake login link",         0.75),
    "financial_prize":      ("Prize / reward lure",     0.85),
    "financial_transfer":   ("Money transfer",          0.70),
    "financial_refund":     ("Fake refund lure",        0.65),
    "financial_loan":       ("Predatory loan offer",    0.60),
    "impersonate_bank":     ("Bank impersonation",      0.85),
    "impersonate_gov":      ("Gov impersonation",       0.85),
    "impersonate_courier":  ("Courier impersonation",   0.75),
    "impersonate_generic":  ("Generic salutation",      0.55),
    "link_shortener":       ("URL shortener",           0.65),
    "link_suspicious_tld":  ("Suspicious TLD",          0.75),
    "link_ip_address":      ("IP-based URL",            0.92),
    "link_login_path":      ("Verify account link",     0.70),
    "social_secret":        ("Secrecy instruction",     0.85),
    "social_download":      ("Force app download",      0.80),
    "social_whatsapp":      ("Messenger redirect",      0.65),
}

_VERDICT_DISPLAY = {
    "clean":      "Clean",
    "suspicious": "Suspicious",
    "high_risk":  "High Risk",
    "scam":       "Scam",
}

_URL_RE = re.compile(
    r"https?://[^\s\)<>\"\']+|www\.[a-z0-9\-]+\.[a-z]{2,}[^\s]*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Campaign matching (mirrors matchCampaign() in the HTML)
# ---------------------------------------------------------------------------
_CAMPAIGN_PATTERNS = [
    ("CR-0047", re.compile(r"maybank|acoount|immediately.*suspend|suspend.*immediately", re.I)),
    ("CR-0039", re.compile(r"lhdn|cukai|bayar|akta\s+cukai", re.I)),
    ("CR-0051", re.compile(r"touch\s*n\s*go|tng.*ewallet|ewallet.*reload", re.I)),
    ("CR-0033", re.compile(r"congratulations|prize|rm\s*50[,.]?000|winner", re.I)),
]


def match_campaign(text: str, score: int) -> Optional[str]:
    if score < 40:
        return None
    for campaign_id, pattern in _CAMPAIGN_PATTERNS:
        if pattern.search(text):
            return campaign_id
    return None


# ---------------------------------------------------------------------------
# Persona matching (mirrors matchPersona() in the HTML)
# ---------------------------------------------------------------------------
_PERSONA_RULES = [
    ("SCP-019", re.compile(r"payment|vendor|supplier|board\s+meeting|approve", re.I)),
    ("SCP-003", re.compile(r"acoount|immediate\s+action\s+required", re.I)),
    ("SCP-022", re.compile(r"lhdn|akta|seksyen|kes\s*:", re.I)),
    ("SCP-012", re.compile(r"congratulations!!!|rm\s*50[,.]?000|do\s+not\s+tell", re.I)),
]


def match_persona(text: str, matched_labels: list[str]) -> Optional[str]:
    has_bank = "Bank impersonation" in matched_labels
    has_suspend = "Account suspension" in matched_labels
    for persona_id, pattern in _PERSONA_RULES:
        if pattern.search(text):
            return persona_id
    if has_bank or has_suspend:
        return "SCP-003"
    return None


# ---------------------------------------------------------------------------
# Singleton Brain loader
# ---------------------------------------------------------------------------
class _BrainHolder:
    def __init__(self) -> None:
        self._brain: Optional[Brain] = None
        self._loaded = False

    def get(self) -> Brain:
        if not self._loaded:
            logger.info("Loading Brain pipeline…")
            self._brain = Brain.load()
            self._loaded = True
            logger.info("Brain pipeline ready.")
        return self._brain  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


_holder = _BrainHolder()


def get_brain() -> Brain:
    return _holder.get()


def layer_status() -> dict[str, bool]:
    if not _holder.is_loaded:
        _holder.get()
    brain = _holder._brain
    ml_ok  = brain is not None and getattr(brain._ml,  "loaded", False)
    nlp_ok = brain is not None and getattr(brain._nlp, "loaded", False)
    return {"rule_engine": True, "ml_model": ml_ok, "nlp_model": nlp_ok}


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------
def analyze(text: str, channel: str, vt_api_key: str = "") -> AnalyzeResponse:
    """
    Run the full cascade and return a structured API response,
    enriched with campaign ID, persona ID, and optional VT results.
    """
    brain = get_brain()

    if channel == "email":
        result: AnalysisResult = brain.analyze_email(body=text)
    elif channel == "url":
        result = brain.analyze_url(text)
    else:
        result = brain.analyze(text, channel="message")

    # Re-run rule engine to capture ALL matched patterns (pipeline caps at 3)
    rule_result = _run_rule_engine(text)
    matched_patterns = _build_matched_patterns(rule_result.matched)
    matched_labels   = [p.label for p in matched_patterns]

    # Extract URLs for VT check
    urls_found = list(dict.fromkeys(_URL_RE.findall(text)))[:10]

    # VirusTotal — only check URLs for suspicious content (score > 30)
    vt_results: list[VTResult] = []
    if urls_found and result.score > 30:
        vt_results = _check_vt(urls_found[:2], vt_api_key)

    # Boost score if VT flagged a URL
    final_score = result.score
    vt_mal_count = sum(1 for r in vt_results if r.mal >= 3)
    if vt_mal_count:
        final_score = min(final_score + 15 * vt_mal_count, 100)

    # Recalculate verdict if score was boosted
    verdict_key = result.verdict
    if final_score != result.score:
        if   final_score <= 30: verdict_key = "clean"
        elif final_score <= 60: verdict_key = "suspicious"
        elif final_score <= 80: verdict_key = "high_risk"
        else:                   verdict_key = "scam"

    campaign_id = match_campaign(text, final_score)
    persona_id  = match_persona(text, matched_labels)

    return AnalyzeResponse(
        analysis_id      = result.analysis_id,
        score            = final_score,
        verdict          = _VERDICT_DISPLAY.get(verdict_key, verdict_key.title()),
        verdict_key      = verdict_key,
        explanation      = result.explanation,
        factors          = result.factors,
        matched_patterns = matched_patterns,
        layers_used      = result.layers_used,
        raw_scores       = RawScores(
            rule = round(result.rule_score, 4),
            ml   = round(result.ml_score,  4) if result.ml_score  is not None else None,
            nlp  = round(result.nlp_score, 4) if result.nlp_score is not None else None,
        ),
        channel          = result.channel,
        latency_ms       = round(result.latency_ms, 2),
        urls_found       = urls_found,
        vt_results       = vt_results,
        campaign_id      = campaign_id,
        persona_id       = persona_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_matched_patterns(raw_labels: list[str]) -> list[MatchedPattern]:
    seen: set[str] = set()
    result: list[MatchedPattern] = []
    for label in raw_labels:
        display, weight = _PATTERN_LABELS.get(
            label, (label.replace("_", " ").title(), 0.60)
        )
        if display not in seen:
            seen.add(display)
            result.append(MatchedPattern(label=display, w=weight))
    result.sort(key=lambda p: p.w, reverse=True)
    return result


def _check_vt(urls: list[str], api_key: str) -> list[VTResult]:
    import concurrent.futures
    from api.virustotal import check_url

    results = []
    # Do NOT use `with` — the context manager calls shutdown(wait=True) on exit,
    # which blocks until the VT thread finishes even after a TimeoutError.
    # Instead we explicitly shut down without waiting so the response is never held.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        futures = {executor.submit(check_url, url, api_key): url for url in urls}
        for future, url in futures.items():
            try:
                raw = future.result(timeout=6)   # 6-second hard cap per URL
                results.append(VTResult(
                    url     = raw["url"],
                    mal     = raw["mal"],
                    sus     = raw["sus"],
                    cln     = raw["cln"],
                    total   = raw.get("total", 72),
                    engines = [VTEngine(n=e["n"], r=e["r"]) for e in raw.get("engines", [])],
                    status  = raw["status"],
                    source  = raw.get("source", "virustotal"),
                ))
            except concurrent.futures.TimeoutError:
                logger.warning("VT check timed out for %s — skipping", url[:50])
            except Exception as exc:
                logger.warning("VT check error for %s: %s", url[:50], exc)
    finally:
        # wait=False: returns immediately; background threads finish on their own
        executor.shutdown(wait=False)
    return results
