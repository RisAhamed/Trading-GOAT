#!/usr/bin/env python3
"""Health check: prints /health-style JSON. Exit 0 if READY/DEGRADED, 2 if NOT_READY."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observability import health

print(json.dumps({"status": health.get_health().status,
                  "components": health.get_health().components,
                  "uptime_s": round(health.get_health().uptime_s, 1)}, indent=2))
sys.exit(0 if health.get_health().status in ("READY", "DEGRADED", "STARTING") else 2)
