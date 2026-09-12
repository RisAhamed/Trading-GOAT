"""DEPRECATED compatibility launcher — canonical runtime is scripts/run_bot.py.

main.py used to be an independent competing trading engine (legacy AITrader).
To guarantee ONE canonical runtime, this file now only forwards to the
canonical loop. The legacy engine is preserved in git history, not on this
path. PAPER TRADING ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

print("NOTE: main.py is deprecated. Forwarding to canonical runtime: scripts/run_bot.py")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv[0] = str(Path(__file__).resolve().parent / "scripts" / "run_bot.py")

from scripts.run_bot import main  # noqa: E402

if __name__ == "__main__":
    main()
