"""Runtime state file: canonical loop publishes, dashboard reads.

Truthful display contract: unknown values are None -> dashboard renders
UNKNOWN / NOT AVAILABLE, never 0-as-missing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path("logs") / "runtime_state.json"


def publish_runtime_state(payload: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            **payload, "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass  # dashboard state is best-effort; never break trading


def read_runtime_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}
