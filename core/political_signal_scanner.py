"""
Political signal scanner using public congressional trade disclosures.
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import requests

from .config_loader import ConfigLoader, get_config


logger = logging.getLogger(__name__)


class PoliticalSignalScanner:
    """Fetches and summarizes political trading signals from public APIs."""

    QUVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config or get_config()
        self._signal_cache: Dict[str, Any] = {"trades": []}
        self._last_fetch_time: float = 0.0
        self._fetch_interval: int = int(
            getattr(
                getattr(self.config, "political_signals", None),
                "fetch_interval_seconds",
                3600,
            )
        )

    def _parse_date(self, value: Any) -> datetime | None:
        """Parse API date values safely."""
        try:
            if not value:
                return None
            if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            text = str(value).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _extract_amount_min(self, amount_text: Any) -> int:
        """Extract lower bound from amount range text."""
        try:
            if amount_text is None:
                return 0
            if isinstance(amount_text, (int, float)):
                return int(amount_text)
            text = str(amount_text).replace(",", "")
            nums = re.findall(r"\d+", text)
            if not nums:
                return 0
            return int(nums[0])
        except Exception:
            return 0

    def fetch_recent_trades(self) -> List[dict]:
        """Fetch and normalize recent congressional trades."""
        try:
            if (
                self._signal_cache.get("trades")
                and (time.time() - self._last_fetch_time) < self._fetch_interval
            ):
                return list(self._signal_cache.get("trades", []))

            headers = {"Accept": "application/json", "X-CSRFToken": ""}
            resp = requests.get(self.QUVER_URL, headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json() or []

            lookback_days = int(
                getattr(
                    getattr(self.config, "political_signals", None),
                    "lookback_days",
                    30,
                )
            )
            min_trade_size = int(
                getattr(
                    getattr(self.config, "political_signals", None),
                    "min_trade_size",
                    1000,
                )
            )
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            normalized: List[dict] = []

            for item in payload:
                try:
                    if not isinstance(item, dict):
                        continue

                    politician = str(
                        item.get("Representative")
                        or item.get("Politician")
                        or item.get("Name")
                        or "Unknown"
                    ).strip()

                    raw_asset = (
                        item.get("Ticker")
                        or item.get("Asset")
                        or item.get("Symbol")
                        or item.get("Security")
                        or ""
                    )
                    asset = str(raw_asset).strip().upper()
                    if "/" in asset:
                        asset = asset.split("/", 1)[0]

                    tx = str(
                        item.get("Transaction")
                        or item.get("Type")
                        or item.get("Action")
                        or ""
                    ).lower()
                    if "buy" in tx or "purchase" in tx:
                        action = "Buy"
                    elif "sell" in tx or "sale" in tx:
                        action = "Sell"
                    else:
                        continue

                    amount_min = self._extract_amount_min(
                        item.get("Range")
                        or item.get("Amount")
                        or item.get("AmountRange")
                        or item.get("AmountMin")
                    )
                    if amount_min < min_trade_size:
                        continue

                    trade_dt = self._parse_date(
                        item.get("TransactionDate")
                        or item.get("Traded")
                        or item.get("TradeDate")
                        or item.get("Date")
                    )
                    disclosure_dt = self._parse_date(
                        item.get("DateRecieved")
                        or item.get("DisclosureDate")
                        or item.get("Filed")
                        or item.get("ReportedDate")
                    )

                    if not trade_dt:
                        continue
                    if trade_dt < cutoff:
                        continue

                    if disclosure_dt:
                        days_lag = max(0, int((disclosure_dt - trade_dt).days))
                    else:
                        days_lag = 0

                    normalized.append(
                        {
                            "politician": politician,
                            "asset": asset,
                            "action": action,
                            "amount_min": int(amount_min),
                            "trade_date": trade_dt.date().isoformat(),
                            "disclosure_date": (
                                disclosure_dt.date().isoformat()
                                if disclosure_dt
                                else ""
                            ),
                            "days_lag": days_lag,
                        }
                    )
                except Exception:
                    continue

            self._signal_cache["trades"] = normalized
            self._last_fetch_time = time.time()
            logger.info(f"[POLITICAL] Loaded {len(normalized)} recent disclosure trades")
            return list(normalized)

        except Exception as e:
            logger.warning(f"[POLITICAL] Failed to fetch recent trades: {e}")
            return list(self._signal_cache.get("trades", []))

    def get_crypto_bias(self, symbol: str) -> float:
        """Return multiplier bias (0.8 - 1.3) from political trading flow."""
        try:
            if not bool(
                getattr(
                    getattr(self.config, "political_signals", None),
                    "apply_to_crypto",
                    True,
                )
            ):
                return 1.0

            asset = str(symbol).split("/", 1)[0].upper()
            trades = self.fetch_recent_trades()

            buy_count = sum(
                1
                for t in trades
                if t.get("asset") == asset and t.get("action") == "Buy"
            )
            sell_count = sum(
                1
                for t in trades
                if t.get("asset") == asset and t.get("action") == "Sell"
            )

            if buy_count >= 3:
                return 1.3
            if buy_count == 2:
                return 1.2
            if buy_count == 1:
                return 1.1
            if sell_count >= 2:
                return 0.8
            return 1.0

        except Exception as e:
            logger.warning(f"[POLITICAL] Bias calculation failed for {symbol}: {e}")
            return 1.0

    def get_top_signals(self) -> List[dict]:
        """Return top 5 disclosed trades by lower amount bound."""
        try:
            trades = self.fetch_recent_trades()
            return sorted(
                trades,
                key=lambda x: int(x.get("amount_min", 0)),
                reverse=True,
            )[:5]
        except Exception as e:
            logger.warning(f"[POLITICAL] Top-signal read failed: {e}")
            return []

    def get_signal_summary(self) -> str:
        """Build one-line UI summary of recent political flow."""
        try:
            trades = self.fetch_recent_trades()
            if not trades:
                return ""

            buys: Dict[str, int] = {}
            sells: Dict[str, int] = {}
            for t in trades:
                asset = str(t.get("asset", "")).upper()
                if not asset:
                    continue
                if t.get("action") == "Buy":
                    buys[asset] = buys.get(asset, 0) + 1
                elif t.get("action") == "Sell":
                    sells[asset] = sells.get(asset, 0) + 1

            parts: List[str] = []
            for asset, count in sorted(buys.items(), key=lambda i: i[1], reverse=True)[:3]:
                parts.append(f"{asset}×{count} {'buy' if count == 1 else 'buys'}")
            for asset, count in sorted(sells.items(), key=lambda i: i[1], reverse=True)[:2]:
                if count > 0:
                    parts.append(f"{asset}×{count} {'sell' if count == 1 else 'sells'}")

            if not parts:
                return ""
            return f"🏛️ Political: {' | '.join(parts)} (last 30d)"

        except Exception:
            return ""
