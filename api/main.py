"""
api/main.py
===========
ShieldAI — FastAPI application.

Endpoints
---------
GET  /                    ShieldAI UI
POST /api/analyze         Full fraud-detection cascade
GET  /api/health          Layer availability
GET  /api/stats           Live dashboard data (KPIs, chart, feed)
GET  /api/history         Paginated analysis history from DB
GET  /api/campaigns       Campaign list enriched with real attack counts
GET  /api/personas        Persona list enriched with real attack counts
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from api.config import get_settings
from api.database import (
    init_db, save_analysis,
    get_recent_analyses, get_stats, get_hourly_data,
    get_threat_breakdown, get_recent_high_risk,
    get_campaign_counts, get_persona_counts, get_last_seen,
)
from api.detector import analyze, layer_status
from api.models import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse, ErrorResponse,
    StatsResponse, ThreatType, RecentDetection,
    HistoryEntry, CampaignData, PersonaData,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shieldai")

_API_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_API_DIR)
_UI_FILE     = os.path.join(_PROJECT_DIR, "ui", "shieldai_v3.html")

_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Campaign & Persona static definitions
# (Counts/last-seen are overridden with live DB values at query time)
# ---------------------------------------------------------------------------
_CAMPAIGNS = [
    {"id":"CR-0047","name":"Maybank Account Suspension Wave","icon":"🏦","color":"var(--red)","variants":23,"firstSeen":"14 days ago","targeting":"Maybank customers, Klang Valley","evolution":"Adding WhatsApp redirect in latest variant","timeline":[4,6,8,5,12,18,22,20,14,9,7,11,16,19,23],"status":"active","statusColor":"var(--red)","desc":"Mass phishing campaign impersonating Maybank security alerts. Evolving to use WhatsApp as a secondary channel to bypass email filters."},
    {"id":"CR-0039","name":"LHDN Tax Penalty Campaign","icon":"🏛","color":"var(--orange)","variants":11,"firstSeen":"31 days ago","targeting":"Malaysian taxpayers, April–May peak","evolution":"Now including QR code redirect","timeline":[2,3,5,8,11,9,7,6,4,3,8,12,10,7,5],"status":"active","statusColor":"var(--orange)","desc":"Government impersonation campaign exploiting tax filing season. High activity in April. Now including QR codes to bypass URL scanners."},
    {"id":"CR-0051","name":"Touch n Go Reload Scam","icon":"💳","color":"var(--purple)","variants":8,"firstSeen":"6 days ago","targeting":"TNG eWallet users, urban Malaysia","evolution":"New — fingerprint still forming","timeline":[0,0,2,5,8,10,12,15,18,14,11,9,13,16,19],"status":"new","statusColor":"var(--accent)","desc":"Emerging campaign targeting Touch n Go users. Uses fake reload bonus offers. Still in early phase — template evolving rapidly."},
    {"id":"CR-0033","name":"Prize & Lottery Blast","icon":"🎁","color":"var(--yellow)","variants":19,"firstSeen":"47 days ago","targeting":"General public, 35–65 age group","evolution":"Template stable — rotating sender domains","timeline":[8,12,15,18,14,11,9,7,5,8,12,14,16,13,11],"status":"active","statusColor":"var(--orange)","desc":"Long-running mass campaign using lottery and prize themes. Template is stable but rotates sender domains every 3 days."},
]

_PERSONAS = [
    {"id":"SCP-003","name":"The Bureaucrat","avatar":"🎭","avatarBg":"var(--red-dim)","avatarBorder":"rgba(220,38,38,.3)","campaigns":4,"firstSeen":"Nov 2023","traits":["Misspells \"acoount\"","Uses CAPS for emphasis","Ends with \"Immediate action required\"","Prefers Bahasa/English mix","Sends 9am–2pm UTC+8"],"targeting":["Maybank","CIMB","LHDN"],"targetBg":"var(--red-dim)","targetBorder":"rgba(220,38,38,.2)","targetColor":"var(--red)","prediction":"Likely to launch new Eid-related campaign within 2 weeks","risk":"HIGH"},
    {"id":"SCP-007","name":"The Impersonator","avatar":"🕵","avatarBg":"var(--purple-dim)","avatarBorder":"rgba(124,58,237,.3)","campaigns":3,"firstSeen":"Jan 2024","traits":["Consistent typo: \"veryfy\"","Uses numbered lists","WhatsApp redirect pattern","Operates evenings UTC+8","Short sentences, high urgency"],"targeting":["PayPal","Amazon","Apple"],"targetBg":"var(--purple-dim)","targetBorder":"rgba(124,58,237,.2)","targetColor":"var(--purple)","prediction":"Evolving to Apple ID scams based on recent variants","risk":"HIGH"},
    {"id":"SCP-012","name":"The Prize Caller","avatar":"🎰","avatarBg":"var(--orange-dim)","avatarBorder":"rgba(217,119,6,.3)","campaigns":2,"firstSeen":"Feb 2024","traits":["Excessive exclamation marks","Always mentions RM 50,000","Uses \"Congratulations!!!\"","Template rarely changes","Operates weekends only"],"targeting":["General public","Grab users"],"targetBg":"var(--orange-dim)","targetBorder":"rgba(217,119,6,.2)","targetColor":"var(--orange)","prediction":"Dormant — last active 3 days ago. May resurface.","risk":"MEDIUM"},
    {"id":"SCP-019","name":"The Corporate","avatar":"💼","avatarBg":"var(--teal-dim)","avatarBorder":"rgba(13,148,136,.3)","campaigns":1,"firstSeen":"Mar 2024","traits":["Professional tone","No typos — high literacy","Always refers to \"urgent payment\"","Uses real company names","Operates business hours only"],"targeting":["Finance teams","HR departments"],"targetBg":"var(--teal-dim)","targetBorder":"rgba(13,148,136,.2)","targetColor":"var(--teal)","prediction":"Specializes in BEC. Likely escalating to larger organizations.","risk":"HIGH"},
    {"id":"SCP-022","name":"The Gov Official","avatar":"🏛","avatarBg":"var(--accent-dim)","avatarBorder":"rgba(29,78,216,.3)","campaigns":2,"firstSeen":"Dec 2023","traits":["Uses official-sounding language","References specific act numbers","Always threatens legal action","Provides fake case numbers","Malay language preferred"],"targeting":["LHDN taxpayers","JPJ users"],"targetBg":"var(--accent-dim)","targetBorder":"rgba(29,78,216,.2)","targetColor":"var(--accent)","prediction":"Tax season activity expected to spike in coming weeks","risk":"HIGH"},
    {"id":"SCP-028","name":"The Parcel Hunter","avatar":"📦","avatarBg":"var(--yellow-dim)","avatarBorder":"rgba(202,138,4,.3)","campaigns":1,"firstSeen":"Mar 2024","traits":["Mentions tracking numbers","Fake PosLaju/GDex branding","Requests RM 2–5 \"clearance fee\"","Short messages only","Uses URL shorteners exclusively"],"targeting":["Online shoppers","PosLaju users"],"targetBg":"var(--yellow-dim)","targetBorder":"rgba(202,138,4,.2)","targetColor":"var(--yellow)","prediction":"Low activity currently. Template unchanged for 2 weeks.","risk":"LOW"},
]

# Feed icon mapping by campaign / channel
_FEED_ICONS = {
    "CR-0047": "🏦", "CR-0039": "🏛", "CR-0033": "🎁",
    "CR-0051": "💳", "email": "📧", "url": "🔗", "message": "💬",
}
_VERDICT_COLOR = {
    "scam":       "var(--red)",
    "high_risk":  "var(--orange)",
    "suspicious": "var(--yellow)",
    "clean":      "var(--green)",
}

# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory, per IP)
# ---------------------------------------------------------------------------
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str, limit: int) -> bool:
    if limit <= 0:
        return True
    now = time.time()
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < 60.0]
    if len(_rate_buckets[ip]) >= limit:
        return False
    _rate_buckets[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ShieldAI starting…")
    init_db()
    try:
        from api.detector import get_brain
        get_brain()
        logger.info("Detection models ready.")
    except Exception as exc:
        logger.warning("Model warm-up issue: %s", exc)
    yield
    logger.info("ShieldAI shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
settings = get_settings()

app = FastAPI(
    title       = "ShieldAI",
    description = "Intelligent Threat Detection Platform",
    version     = "1.0.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Content-Type"],
    allow_credentials = False,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_ui():
    if not os.path.isfile(_UI_FILE):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="UI file not found.")
    return FileResponse(_UI_FILE, media_type="text/html")


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    summary="Analyze text / email / URL for fraud indicators",
)
def analyze_endpoint(request: Request, body: AnalyzeRequest):
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip, settings.rate_limit):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded.")
    try:
        result = analyze(
            text      = body.input,
            channel   = body.type,
            vt_api_key= settings.vt_api_key,
        )
    except Exception as exc:
        logger.error("Analysis error: %s", exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Analysis failed due to an internal error.")

    # Persist to DB
    save_analysis({
        "id":              result.analysis_id,
        "score":           result.score,
        "verdict":         result.verdict,
        "verdict_key":     result.verdict_key,
        "channel":         result.channel,
        "matched_patterns":[{"label": p.label, "w": p.w}
                            for p in result.matched_patterns],
        "factors":         result.factors,
        "layers_used":     result.layers_used,
        "rule_score":      result.raw_scores.rule,
        "ml_score":        result.raw_scores.ml,
        "nlp_score":       result.raw_scores.nlp,
        "latency_ms":      result.latency_ms,
        "input_preview":   body.input[:120],
        "campaign_id":     result.campaign_id,
        "persona_id":      result.persona_id,
        "created_at":      datetime.now(timezone.utc).isoformat(),
    })

    logger.info("Analysis %s | score=%d | verdict=%s | channel=%s | %.1fms",
                result.analysis_id, result.score, result.verdict_key,
                result.channel, result.latency_ms)
    return result


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", layers=layer_status(), version="1.0.0")


@app.get("/api/stats", response_model=StatsResponse, summary="Live dashboard data")
def stats_endpoint():
    """Returns KPIs, hourly chart data, threat breakdown, and recent detections."""
    db_stats      = get_stats()
    hourly        = get_hourly_data(24)
    threat_types  = get_threat_breakdown()
    recent_raw    = get_recent_high_risk(5)

    total    = db_stats["total"]
    blocked  = db_stats["blocked"]
    det_rate = round(blocked / total * 100, 1) if total > 0 else 0.0

    # Build recent detections feed
    recent_detections = []
    for row in recent_raw:
        camp_id = row.get("campaign_id")
        icon    = _FEED_ICONS.get(camp_id) or _FEED_ICONS.get(row["channel"], "💬")
        layers  = json.loads(row.get("layers_json", "[]"))
        layer_str = " · ".join(l.upper() for l in layers) if layers else "RULE"
        lat   = row.get("latency_ms") or 0
        title = row["input_preview"][:60] or "(no preview)"
        sub   = f"{row['channel'].upper()} · {layer_str} · {round(lat)}ms"
        if camp_id:
            camp_name = next((c["name"] for c in _CAMPAIGNS if c["id"] == camp_id), camp_id)
            sub += f" · {camp_id}"
            title = f"[{camp_id}] {title}"

        recent_detections.append(RecentDetection(
            id    = row["id"],
            icon  = icon,
            title = title,
            sub   = sub,
            score = row["score"],
            color = _VERDICT_COLOR.get(row["verdict_key"], "var(--text)"),
            time  = _relative_time(row["created_at"]),
        ))

    return StatsResponse(
        total_analyzed   = total,
        threats_blocked  = blocked,
        scam_count       = db_stats["scam"],
        high_risk_count  = db_stats["high_risk"],
        suspicious_count = db_stats["suspicious"],
        clean_count      = db_stats["clean"],
        campaigns_active = len(_CAMPAIGNS),
        detection_rate   = det_rate,
        by_channel       = db_stats["by_channel"],
        hourly_data      = hourly,
        threat_types     = [ThreatType(**t) for t in threat_types],
        recent_detections= recent_detections,
        uptime_seconds   = int(time.time() - _START_TIME),
    )


@app.get("/api/history", summary="Analysis history from database")
def history_endpoint(limit: int = Query(default=50, ge=1, le=200)):
    rows = get_recent_analyses(limit)
    entries = []
    for row in rows:
        try:
            layers = json.loads(row.get("layers_json", "[]"))
        except Exception:
            layers = []
        entries.append(HistoryEntry(
            id            = row["id"],
            score         = row["score"],
            verdict       = row["verdict"],
            verdict_key   = row["verdict_key"],
            channel       = row["channel"],
            input_preview = row["input_preview"],
            campaign_id   = row.get("campaign_id"),
            persona_id    = row.get("persona_id"),
            layers_used   = layers,
            latency_ms    = row.get("latency_ms"),
            created_at    = row["created_at"],
            time_label    = _relative_time(row["created_at"]),
        ))
    return entries


@app.get("/api/campaigns", summary="Campaign list with live attack counts")
def campaigns_endpoint():
    counts   = get_campaign_counts()
    result   = []
    for camp in _CAMPAIGNS:
        cid         = camp["id"]
        live_count  = counts.get(cid, 0)
        last_seen   = get_last_seen("campaign_id", cid) or camp.get("firstSeen", "–")
        result.append(CampaignData(
            **camp,
            attacks  = live_count,
            lastSeen = last_seen,
        ))
    return result


@app.get("/api/personas", summary="Persona list with live attack counts")
def personas_endpoint():
    counts = get_persona_counts()
    result = []
    for persona in _PERSONAS:
        pid        = persona["id"]
        live_count = counts.get(pid, 0)
        last_seen  = get_last_seen("persona_id", pid) or "No recent activity"
        active     = last_seen not in ("No recent activity",) and \
                     not last_seen.endswith("d ago") or live_count > 0
        result.append(PersonaData(
            **persona,
            attacks  = live_count,
            lastSeen = last_seen,
            active   = active,
        ))
    return result


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValidationError)
async def pydantic_handler(request: Request, exc: ValidationError):
    return JSONResponse(status.HTTP_400_BAD_REQUEST,
                        {"detail": exc.errors()[0]["msg"] if exc.errors()
                                   else "Validation error"})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status.HTTP_500_INTERNAL_SERVER_ERROR,
                        {"detail": "An unexpected error occurred."})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _relative_time(iso: str) -> str:
    try:
        ts    = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        secs  = int((datetime.now(timezone.utc) - ts).total_seconds())
        if secs < 60:   return "just now"
        if secs < 3600: return f"{secs // 60}m ago"
        if secs < 86400:return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "–"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.host, port=settings.port,
                reload=settings.reload)
