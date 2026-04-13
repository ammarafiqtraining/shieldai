"""
api/virustotal.py
=================
Real VirusTotal API v3 integration.

Flow per URL:
  1. Try to fetch the existing URL report (cached in VT's database).
     Most URLs that look suspicious have already been scanned.
  2. If not found (404), submit the URL for scanning and poll once.
  3. Cache the result locally for 24 hours to stay within rate limits.

Rate limits (free tier):  4 requests / minute.
We are conservative: 1 request per 20 seconds max.

If VT_API_KEY is empty the module returns mock/demo results so the UI
still renders cleanly without a real key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory — shared with the data volume so it survives restarts
_CACHE_DIR = Path(os.environ.get("DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "shieldai.db")
)).parent / "vt_cache"

_CACHE_TTL_HOURS = 24
_VT_BASE = "https://www.virustotal.com/api/v3"
_REQUEST_TIMEOUT = 8          # seconds per HTTP call
_MIN_INTERVAL = 20.0          # seconds between VT calls (3 req/min, conservative)

_last_call_time: float = 0.0  # module-level rate limiter


def check_url(url: str, api_key: str) -> dict:
    """
    Check a URL against VirusTotal.

    Returns a dict compatible with the UI's VT card renderer:
    {
        "url":      str,
        "mal":      int,   # malicious engine count
        "sus":      int,   # suspicious engine count
        "cln":      int,   # clean engine count
        "total":    int,   # total engines
        "engines":  list[{"n": name, "r": result}],
        "status":   "malicious" | "suspicious" | "clean",
        "source":   "virustotal" | "demo"
    }
    """
    if not api_key:
        return _demo_result(url)

    # Check local cache first
    cached = _get_cache(url)
    if cached is not None:
        return cached

    # Rate limit — never hit VT faster than _MIN_INTERVAL
    _rate_limit()

    try:
        result = _fetch_url_report(url, api_key)
    except Exception as exc:
        logger.warning("VT check failed for %s: %s", url[:60], exc)
        return _demo_result(url)

    _save_cache(url, result)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _url_id(url: str) -> str:
    """VirusTotal URL ID = base64url(url) without padding."""
    return base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()


def _rate_limit() -> None:
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.time()


def _fetch_url_report(url: str, api_key: str) -> dict:
    """Fetch the VT URL report. Submits for scanning if not found."""
    import urllib.request
    import urllib.error
    import urllib.parse

    headers = {"x-apikey": api_key, "Accept": "application/json"}

    def _request(method: str, path: str, data: bytes | None = None) -> dict:
        req = urllib.request.Request(
            f"{_VT_BASE}{path}",
            data=data,
            headers={**headers,
                     **({"Content-Type": "application/x-www-form-urlencoded"}
                        if data else {})},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read())

    # Try existing report first
    try:
        data = _request("GET", f"/urls/{_url_id(url)}")
        return _parse_url_report(url, data)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    # Not in VT — submit for scanning (no extra rate_limit; caller already waited)
    payload = urllib.parse.urlencode({"url": url}).encode()
    submit = _request("POST", "/urls", data=payload)
    analysis_id = submit.get("data", {}).get("id", "")

    if not analysis_id:
        raise ValueError("No analysis ID returned from VT submission")

    # Brief fixed delay — VT needs ~2 s to queue the scan; no full rate-limit sleep
    time.sleep(2)
    result = _request("GET", f"/analyses/{analysis_id}")
    return _parse_analysis_result(url, result)


def _parse_url_report(url: str, data: dict) -> dict:
    stats = (data.get("data", {})
                 .get("attributes", {})
                 .get("last_analysis_stats", {}))
    results = (data.get("data", {})
                   .get("attributes", {})
                   .get("last_analysis_results", {}))
    return _build_result(url, stats, results)


def _parse_analysis_result(url: str, data: dict) -> dict:
    stats = (data.get("data", {})
                 .get("attributes", {})
                 .get("stats", {}))
    results = (data.get("data", {})
                   .get("attributes", {})
                   .get("results", {}))
    return _build_result(url, stats, results)


def _build_result(url: str, stats: dict, results: dict) -> dict:
    mal   = stats.get("malicious", 0)
    sus   = stats.get("suspicious", 0)
    cln   = stats.get("harmless", 0) + stats.get("undetected", 0)
    total = mal + sus + cln + stats.get("timeout", 0)

    # Extract a representative sample of engine results for display
    engines = []
    for engine_name, res in list(results.items())[:12]:
        cat = res.get("category", res.get("result", "clean"))
        engines.append({"n": engine_name, "r": cat})

    if mal >= 3:
        status = "malicious"
    elif sus >= 3 or mal >= 1:
        status = "suspicious"
    else:
        status = "clean"

    return {
        "url":     url,
        "mal":     mal,
        "sus":     sus,
        "cln":     cln,
        "total":   total or 72,
        "engines": engines,
        "status":  status,
        "source":  "virustotal",
    }


# ---------------------------------------------------------------------------
# File-based cache
# ---------------------------------------------------------------------------

def _cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    return _CACHE_DIR / f"{key}.json"


def _get_cache(url: str) -> Optional[dict]:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(payload["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=_CACHE_TTL_HOURS):
            path.unlink(missing_ok=True)
            return None
        return payload["result"]
    except Exception:
        return None


def _save_cache(url: str, result: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(url).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "result":    result,
        }))
    except Exception as exc:
        logger.debug("VT cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Demo / fallback results (used when no API key is set)
# ---------------------------------------------------------------------------

_SUSPICIOUS_PATTERNS = (
    r"(\d{1,3}\.){3}\d{1,3}",   # raw IP
    r"bit\.ly|tinyurl",           # shorteners
    r"\.(xyz|top|click|loan)",    # sus TLDs
    r"paypa1|maybank-verify|lhdn-semak",  # typosquatting
)


def _demo_result(url: str) -> dict:
    """Return a plausible demo result based on URL heuristics."""
    import re
    is_bad = any(re.search(p, url, re.IGNORECASE) for p in _SUSPICIOUS_PATTERNS)
    mal = (hash(url) % 18 + 8) if is_bad else (hash(url) % 3)
    sus = (hash(url) % 5 + 1)  if is_bad else (hash(url) % 2)
    cln = max(72 - mal - sus, 0)
    engines = [
        {"n": "Google Safe Browsing", "r": "phishing"    if is_bad and mal > 5  else "clean"},
        {"n": "Kaspersky",            "r": "malicious"   if is_bad and mal > 3  else "clean"},
        {"n": "ESET",                 "r": "phishing"    if is_bad and mal > 8  else "clean"},
        {"n": "Symantec",             "r": "malware"     if is_bad and mal > 10 else "clean"},
        {"n": "Fortinet",             "r": "phishing"    if is_bad and mal > 6  else "clean"},
        {"n": "Sophos",               "r": "suspicious"  if is_bad and mal > 4  else "clean"},
    ]
    status = "malicious" if mal >= 3 else ("suspicious" if sus >= 3 else "clean")
    return {"url": url, "mal": mal, "sus": sus, "cln": cln,
            "total": 72, "engines": engines, "status": status, "source": "demo"}
