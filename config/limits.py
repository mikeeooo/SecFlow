"""MIA Plăți Instant limits, read from `config/policy.json` (`mia_limits`).

One source of truth for the generator (which never creates a payment MIA would refuse), the
feature pipeline (cap-ratio features) and the API (which rejects personal payments above the
per-transaction cap). Amounts are in MDL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "config" / "policy.json"


@dataclass(frozen=True)
class SegmentLimits:
    per_transaction: float
    per_day: float
    per_month: float | None


@dataclass(frozen=True)
class MiaLimits:
    currency: str
    personal: SegmentLimits
    business: SegmentLimits
    fee_free_per_month: float

    def for_segment(self, segment: str) -> SegmentLimits:
        return self.business if segment == "business" else self.personal


def parse_limits(raw: dict) -> MiaLimits:
    def segment(block: dict) -> SegmentLimits:
        month = block.get("per_month")
        return SegmentLimits(float(block["per_transaction"]), float(block["per_day"]),
                             float(month) if month is not None else None)

    limits = MiaLimits(raw["currency"], segment(raw["personal"]), segment(raw["business"]),
                       float(raw["fee_free_per_month"]))
    if limits.currency != "MDL":
        raise ValueError("MIA limits must be expressed in MDL")
    return limits


@lru_cache(maxsize=4)
def load_limits(path: Path = POLICY_PATH) -> MiaLimits:
    return parse_limits(json.loads(Path(path).read_text(encoding="utf-8"))["mia_limits"])


MIA_LIMITS = load_limits()
