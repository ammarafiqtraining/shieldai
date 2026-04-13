"""
api/database.py
===============
SQLite persistence layer for ShieldAI analysis results.

All analyses are stored here so the dashboard, history, campaign tracker,
and persona pages show real accumulated data instead of demo values.

The DB file lives at /app/data/shieldai.db inside the container, which is
mounted as a named Docker volume — data survives container restarts.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# DB path — can be overridden via DB_PATH env var
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "shieldai.db",
)
_DB_PATH = os.environ.get("DB_PATH", _DEFAULT_DB)

# Thread-local storage so each thread gets its own connection
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")    # better concurrent reads
        _local.conn.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id            TEXT    PRIMARY KEY,
            score         INTEGER NOT NULL,
            verdict       TEXT    NOT NULL,
            verdict_key   TEXT    NOT NULL,
            channel       TEXT    NOT NULL,
            matched_json  TEXT    NOT NULL DEFAULT '[]',
            factors_json  TEXT    NOT NULL DEFAULT '[]',
            layers_json   TEXT    NOT NULL DEFAULT '[]',
            rule_score    REAL,
            ml_score      REAL,
            nlp_score     REAL,
            latency_ms    REAL,
            input_preview TEXT    NOT NULL DEFAULT '',
            campaign_id   TEXT,
            persona_id    TEXT,
            created_at    TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_created_at  ON analyses(created_at);
        CREATE INDEX IF NOT EXISTS idx_verdict_key ON analyses(verdict_key);
        CREATE INDEX IF NOT EXISTS idx_campaign_id ON analyses(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_persona_id  ON analyses(persona_id);
    """)
    conn.commit()
    logger.info("Database ready at %s", _DB_PATH)


def save_analysis(record: dict[str, Any]) -> None:
    """Persist one analysis result. Silently skips on error (non-critical)."""
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT OR IGNORE INTO analyses
                (id, score, verdict, verdict_key, channel,
                 matched_json, factors_json, layers_json,
                 rule_score, ml_score, nlp_score, latency_ms,
                 input_preview, campaign_id, persona_id, created_at)
            VALUES
                (:id, :score, :verdict, :verdict_key, :channel,
                 :matched_json, :factors_json, :layers_json,
                 :rule_score, :ml_score, :nlp_score, :latency_ms,
                 :input_preview, :campaign_id, :persona_id, :created_at)
        """, {
            "id":            record["id"],
            "score":         record["score"],
            "verdict":       record["verdict"],
            "verdict_key":   record["verdict_key"],
            "channel":       record["channel"],
            "matched_json":  json.dumps(record.get("matched_patterns", [])),
            "factors_json":  json.dumps(record.get("factors", [])),
            "layers_json":   json.dumps(record.get("layers_used", [])),
            "rule_score":    record.get("rule_score"),
            "ml_score":      record.get("ml_score"),
            "nlp_score":     record.get("nlp_score"),
            "latency_ms":    record.get("latency_ms"),
            "input_preview": record.get("input_preview", "")[:120],
            "campaign_id":   record.get("campaign_id"),
            "persona_id":    record.get("persona_id"),
            "created_at":    record.get("created_at",
                             datetime.now(timezone.utc).isoformat()),
        })
        conn.commit()
    except Exception as exc:
        logger.error("Failed to save analysis: %s", exc)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_recent_analyses(limit: int = 50) -> list[dict]:
    """Return most recent analyses for the History tab."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, score, verdict, verdict_key, channel,
               matched_json, layers_json, latency_ms,
               input_preview, campaign_id, persona_id, created_at
        FROM   analyses
        ORDER  BY created_at DESC
        LIMIT  ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict[str, Any]:
    """Aggregate stats for the dashboard KPIs."""
    conn = _get_conn()

    row = conn.execute("""
        SELECT
            COUNT(*)                                           AS total,
            SUM(CASE WHEN verdict_key IN ('high_risk','scam')
                     THEN 1 ELSE 0 END)                        AS blocked,
            SUM(CASE WHEN verdict_key = 'scam'     THEN 1 ELSE 0 END) AS scam,
            SUM(CASE WHEN verdict_key = 'high_risk' THEN 1 ELSE 0 END) AS high_risk,
            SUM(CASE WHEN verdict_key = 'suspicious' THEN 1 ELSE 0 END) AS suspicious,
            SUM(CASE WHEN verdict_key = 'clean'    THEN 1 ELSE 0 END) AS clean
        FROM analyses
    """).fetchone()

    by_channel = {}
    for r in conn.execute(
        "SELECT channel, COUNT(*) AS n FROM analyses GROUP BY channel"
    ).fetchall():
        by_channel[r["channel"]] = r["n"]

    return {
        "total":      row["total"]   or 0,
        "blocked":    row["blocked"] or 0,
        "scam":       row["scam"]    or 0,
        "high_risk":  row["high_risk"] or 0,
        "suspicious": row["suspicious"] or 0,
        "clean":      row["clean"]   or 0,
        "by_channel": by_channel,
    }


def get_hourly_data(hours: int = 24) -> list[int]:
    """Return analysis counts for the last N hours (oldest → newest)."""
    conn = _get_conn()
    rows = conn.execute("""
        WITH RECURSIVE hours(h) AS (
            SELECT 0
            UNION ALL
            SELECT h + 1 FROM hours WHERE h < ?
        )
        SELECT
            h,
            COUNT(a.id) AS cnt
        FROM hours
        LEFT JOIN analyses a
            ON  strftime('%Y-%m-%dT%H', a.created_at) =
                strftime('%Y-%m-%dT%H',
                    datetime('now', '-' || (? - h) || ' hours'))
        GROUP BY h
        ORDER BY h
    """, (hours - 1, hours - 1)).fetchall()
    return [r["cnt"] for r in rows]


def get_threat_breakdown() -> list[dict]:
    """Return top pattern labels with counts for the dashboard threat-types panel."""
    conn = _get_conn()
    # Extract labels from the JSON array stored in matched_json
    rows = conn.execute("""
        SELECT matched_json FROM analyses
        WHERE  verdict_key IN ('high_risk','scam','suspicious')
        ORDER  BY created_at DESC
        LIMIT  500
    """).fetchall()

    counts: dict[str, int] = {}
    for row in rows:
        try:
            patterns = json.loads(row["matched_json"])
            for p in patterns:
                label = p.get("label", "") if isinstance(p, dict) else str(p)
                if label:
                    counts[label] = counts.get(label, 0) + 1
        except Exception:
            pass

    total = sum(counts.values()) or 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]
    return [
        {"name": label, "count": cnt, "percentage": round(cnt / total * 100)}
        for label, cnt in sorted_items
    ]


def get_recent_high_risk(limit: int = 5) -> list[dict]:
    """Return recent high-risk and scam detections for the dashboard feed."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, score, verdict, verdict_key, channel,
               campaign_id, persona_id, layers_json,
               latency_ms, input_preview, created_at
        FROM   analyses
        WHERE  verdict_key IN ('high_risk', 'scam')
        ORDER  BY created_at DESC
        LIMIT  ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_campaign_counts() -> dict[str, int]:
    """Return match counts per campaign ID."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT campaign_id, COUNT(*) AS n
        FROM   analyses
        WHERE  campaign_id IS NOT NULL
        GROUP  BY campaign_id
    """).fetchall()
    return {r["campaign_id"]: r["n"] for r in rows}


def get_persona_counts() -> dict[str, int]:
    """Return match counts per persona ID."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT persona_id, COUNT(*) AS n
        FROM   analyses
        WHERE  persona_id IS NOT NULL
        GROUP  BY persona_id
    """).fetchall()
    return {r["persona_id"]: r["n"] for r in rows}


def get_last_seen(id_field: str, id_value: str) -> str | None:
    """Return the most recent analysis timestamp for a campaign or persona."""
    conn = _get_conn()
    row = conn.execute(
        f"SELECT created_at FROM analyses WHERE {id_field} = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (id_value,)
    ).fetchone()
    if not row:
        return None
    # Convert ISO timestamp → relative label
    try:
        ts = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 120:    return "just now"
        if secs < 3600:   return f"{secs // 60}m ago"
        if secs < 86400:  return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return None
