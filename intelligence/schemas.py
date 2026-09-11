"""Strict schema for LLM output. Invalid -> HOLD. Never trades malformed output."""
from __future__ import annotations

import json
from dataclasses import dataclass

from domain.models import Action

VALID_ACTIONS = {"BUY", "SELL", "HOLD", "CLOSE"}
SCHEMA_VERSION = "1"


@dataclass
class ParsedAI:
    action: Action
    confidence: float
    trend: str
    entry_quality: str
    reasoning_summary: str
    risk_notes: list[str]
    raw_invalid: bool = False
    failure_reason: str = ""


def _coerce_confidence(v) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return -1.0
    if c > 1.0:  # tolerate 0-100 scale
        c = c / 100.0
    return c


def parse_ai_json(text: str) -> ParsedAI:
    """Parse + validate. Returns raw_invalid=True instead of raising for trade path."""
    if not text or not text.strip():
        return ParsedAI(Action.HOLD, 0.0, "UNKNOWN", "UNKNOWN", "", [], True, "empty_response")
    try:
        # tolerate code fences
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            nl = t.find("\n")
            if nl != -1:
                t = t[nl + 1:]
        data = json.loads(t)
    except Exception as e:
        return ParsedAI(Action.HOLD, 0.0, "UNKNOWN", "UNKNOWN", "", [], True, f"json_parse_error: {e}")
    action = str(data.get("action", "HOLD")).upper()
    if action not in VALID_ACTIONS:
        return ParsedAI(Action.HOLD, 0.0, "UNKNOWN", "UNKNOWN", "", [], True, f"bad_action: {action!r}")
    conf = _coerce_confidence(data.get("confidence", -1))
    if not (0.0 <= conf <= 1.0):
        return ParsedAI(Action.HOLD, 0.0, "UNKNOWN", "UNKNOWN", "", [], True, f"bad_confidence: {data.get('confidence')!r}")
    summary = str(data.get("reasoning_summary", ""))[:500]
    notes = data.get("risk_notes", []) or []
    notes = [str(n)[:200] for n in notes][:5]
    return ParsedAI(
        action=Action(action),
        confidence=conf,
        trend=str(data.get("trend", "UNKNOWN"))[:20],
        entry_quality=str(data.get("entry_quality", "UNKNOWN"))[:20],
        reasoning_summary=summary,
        risk_notes=notes,
    )
