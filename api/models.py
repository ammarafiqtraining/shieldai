"""
api/models.py
=============
Pydantic request / response schemas — single source of truth for
everything the API accepts and returns.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator

_MAX_INPUT_LEN = 10_000
_MIN_INPUT_LEN = 2


# ── Request ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    input: str
    type: Literal["text", "email", "url"] = "text"

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        v = v.strip()
        if len(v) < _MIN_INPUT_LEN:
            raise ValueError(f"Input must be at least {_MIN_INPUT_LEN} characters.")
        if len(v) > _MAX_INPUT_LEN:
            raise ValueError(f"Input exceeds {_MAX_INPUT_LEN:,}-character limit.")
        return v


# ── Shared sub-types ─────────────────────────────────────────────────────────

class MatchedPattern(BaseModel):
    label: str
    w: float  # weight 0–1, maps directly to the UI's RULES format


class RawScores(BaseModel):
    rule: float
    ml:  Optional[float] = None
    nlp: Optional[float] = None


class VTEngine(BaseModel):
    n: str   # engine name
    r: str   # result label


class VTResult(BaseModel):
    url:     str
    mal:     int    # malicious count
    sus:     int    # suspicious count
    cln:     int    # clean count
    total:   int    # total engines
    engines: list[VTEngine]
    status:  str    # "malicious" | "suspicious" | "clean"
    source:  str    # "virustotal" | "demo"


# ── Main analysis response ────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    analysis_id:      str
    score:            int
    verdict:          str          # "Clean" | "Suspicious" | "High Risk" | "Scam"
    verdict_key:      str          # "clean" | "suspicious" | "high_risk" | "scam"
    explanation:      str
    factors:          list[str]
    matched_patterns: list[MatchedPattern]
    layers_used:      list[str]
    raw_scores:       RawScores
    channel:          str
    latency_ms:       float
    urls_found:       list[str]
    vt_results:       list[VTResult] = []
    campaign_id:      Optional[str] = None
    persona_id:       Optional[str] = None


# ── Dashboard / stats ─────────────────────────────────────────────────────────

class ThreatType(BaseModel):
    name:       str
    count:      int
    percentage: int


class RecentDetection(BaseModel):
    id:       str
    icon:     str
    title:    str
    sub:      str
    score:    int
    color:    str
    time:     str


class HourlyBar(BaseModel):
    hour:  int    # 0-23
    count: int


class StatsResponse(BaseModel):
    total_analyzed:    int
    threats_blocked:   int
    scam_count:        int
    high_risk_count:   int
    suspicious_count:  int
    clean_count:       int
    campaigns_active:  int
    detection_rate:    float
    by_channel:        dict[str, int]
    hourly_data:       list[int]          # 24 values, oldest → newest
    threat_types:      list[ThreatType]
    recent_detections: list[RecentDetection]
    uptime_seconds:    int


# ── History ───────────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    id:           str
    score:        int
    verdict:      str
    verdict_key:  str
    channel:      str
    input_preview: str
    campaign_id:  Optional[str]
    persona_id:   Optional[str]
    layers_used:  list[str]
    latency_ms:   Optional[float]
    created_at:   str
    time_label:   str             # human-relative, e.g. "2m ago"


# ── Campaigns ─────────────────────────────────────────────────────────────────

class CampaignData(BaseModel):
    id:           str
    name:         str
    icon:         str
    color:        str
    variants:     int
    attacks:      int             # real count from DB
    firstSeen:    str
    lastSeen:     str             # real last-seen from DB (or static if no data)
    targeting:    str
    evolution:    str
    timeline:     list[int]
    status:       str
    statusColor:  str
    desc:         str


# ── Personas ──────────────────────────────────────────────────────────────────

class PersonaData(BaseModel):
    id:           str
    name:         str
    avatar:       str
    avatarBg:     str
    avatarBorder: str
    attacks:      int             # real count from DB
    campaigns:    int
    firstSeen:    str
    lastSeen:     str             # real last-seen from DB
    active:       bool
    traits:       list[str]
    targeting:    list[str]
    targetBg:     str
    targetBorder: str
    targetColor:  str
    prediction:   str
    risk:         str


# ── Health / error ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    layers:  dict[str, bool]
    version: str


class ErrorResponse(BaseModel):
    detail: str
