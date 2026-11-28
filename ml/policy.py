"""Rules, the risk score and the decision table, all configured in `config/policy.json`.

risk = 100 x (w_IF x IF + w_LOF x LOF + w_DBSCAN x DBSCAN + w_rules x rules), with fixed weights that
sum to 1, so every score splits into per-component contributions. The decision then follows a fixed
priority order: trusted evidence, safeguards, corroborated Block, Verify, Permit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.limits import MiaLimits, parse_limits
from ml.evidence import SecurityEvidence

COMPONENTS = ("isolation_forest", "lof", "dbscan", "rules")
SEGMENTS = ("personal", "business")
OPERATING_POINT_FILE = "operating_point.json"


def segment_of(account: Any) -> str:
    """Sender segment; unknown accounts count as personal."""
    return "business" if getattr(account, "account_type", None) == "business" else "personal"


@dataclass(frozen=True)
class RuleSpec:
    code: str
    family: str
    weight: float
    corroborating: bool
    description: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Policy:
    version: str
    weights: Mapping[str, float]
    initial_verify: float
    initial_block: float
    threshold_search: Mapping[str, Mapping[str, float]]
    goals: Mapping[str, float]
    new_account_verify_reserve: float
    min_families: int
    families: Mapping[str, str]
    rules: tuple[RuleSpec, ...]
    cold_start_max_amount: float
    strong_auth_methods: frozenset[str]
    trusted_sources: tuple[str, ...]
    blocking_types: tuple[str, ...]
    limits: MiaLimits


def load_policy(path: Path) -> Policy:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    weights = {name: float(raw["weights"][name]) for name in COMPONENTS}
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Policy weights must sum to 1")
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return Policy(
        version=f"{raw['policy_name']}-{raw['policy_revision']}+{digest}",
        weights=weights,
        initial_verify=float(raw["initial_thresholds"]["verify"]),
        initial_block=float(raw["initial_thresholds"]["block"]),
        threshold_search=raw["threshold_search"],
        goals=raw["goals"],
        new_account_verify_reserve=float(raw["new_account_verify_reserve"]),
        min_families=int(raw["corroboration"]["min_distinct_families"]),
        families=raw["corroboration"]["families"],
        rules=tuple(RuleSpec(r["code"], r["family"], float(r["weight"]), bool(r["corroborating"]), r["description"],
                             r["params"]) for r in raw["rules"]),
        cold_start_max_amount=float(raw["cold_start"]["max_amount_without_history"]),
        strong_auth_methods=frozenset(raw["cold_start"]["strong_auth_methods"]),
        trusted_sources=tuple(raw["evidence"]["trusted_sources"]),
        blocking_types=tuple(raw["evidence"]["blocking_types"]),
        limits=parse_limits(raw["mia_limits"]),
    )


@dataclass(frozen=True)
class OperatingPoint:
    """Verify and Block thresholds chosen on validation (the same pair for both segments)."""

    verify: float
    block: float
    source: str
    version: str

    @property
    def segments(self) -> dict[str, tuple[float, float]]:
        return {segment: (self.verify, self.block) for segment in SEGMENTS}


def load_operating_point(directory: Path, model_version: str) -> OperatingPoint:
    raw = json.loads((directory / OPERATING_POINT_FILE).read_text())
    if raw["model_version"] != model_version:
        raise ValueError("The operating point was chosen for another model version; retrain.")
    return OperatingPoint(float(raw["verify_threshold"]), float(raw["block_threshold"]), raw["source"], raw["version"])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggeredRule:
    code: str
    family: str
    weight: float
    corroborating: bool
    description: str
    detail: str


@dataclass(frozen=True)
class RuleResult:
    score: float  # min(1, sum of the triggered rules' weights)
    triggered: tuple[TriggeredRule, ...]
    corroborating_families: tuple[str, ...]  # distinct families of the triggered corroborating rules


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f} MDL"


def _rule_check(rule: RuleSpec, f: Mapping[str, Any], has_history: bool, fired: set[str]) -> str | None:
    """A human-readable detail when the rule fires, otherwise None."""
    p = rule.params
    z = f.get("amount_zscore_30d")
    recent_beneficiary = f.get("recipient_history_days") is None or f["recipient_history_days"] <= p.get(
        "max_recipient_history_days", 7)
    code = rule.code
    if code == "AMOUNT_SPIKE":
        if not has_history or f.get("prior_requests_30d", 0) < 3:
            return None
        ratio = f["amount"] / f["max_amount_30d"] if f.get("max_amount_30d") else 0.0
        if (z is not None and z >= p["min_zscore"]) or ratio >= p["min_ratio_to_prior_max"]:
            return (f"{_money(f['amount'])} versus a 30-day median of {_money(f.get('median_amount_30d'))} "
                    f"(z-score {z if z is not None else 'n/a'}, {ratio:.1f}x the prior 30-day maximum)")
    elif code == "AMOUNT_ELEVATED":
        if has_history and "AMOUNT_SPIKE" not in fired and z is not None and z >= p["min_zscore"]:
            return f"z-score {z} against the sender's 30-day amounts"
    elif code == "VELOCITY_BURST":
        ratio = f.get("velocity_ratio_24h", 0.0)
        if (f["requests_1h"] >= p["min_requests_1h"] and ratio >= p["min_velocity_ratio"] / 2) or (
                f["requests_24h"] >= p["min_requests_24h"] and ratio >= p["min_velocity_ratio"]):
            return f"{f['requests_1h']} payments in the last hour, {f['requests_24h']} in 24 hours ({ratio:.1f}x usual pace)"
    elif code == "RECIPIENT_FANOUT":
        if (f["unique_recipients_24h"] >= p["min_unique_recipients_24h"]
                and f.get("velocity_ratio_24h", 0) >= p["min_velocity_ratio"]):
            return f"{f['unique_recipients_24h']} different recipients in 24 hours"
    elif code == "NEW_DEVICE_AFTER_FAILED_LOGINS":
        if has_history and f.get("device_change") and f["auth_failures_24h"] >= p["min_failures_24h"]:
            return (f"device first seen {f.get('device_age_hours') or 0:.1f} h ago, "
                    f"{f['auth_failures_24h']} failed sign-ins in 24 h")
    elif code == "REPEATED_FAILED_LOGINS":
        if f["auth_failures_24h"] >= p["min_failures_24h"]:
            return f"{f['auth_failures_24h']} failed sign-in attempts in 24 hours"
    elif code == "NEW_DEVICE":
        if has_history and f.get("device_change"):
            return f"device first seen {f.get('device_age_hours') or 0:.1f} hours ago"
    elif code == "FOREIGN_SESSION":
        if f["foreign_ip"]:
            return f"session from {f['ip_country']}, home country {f['home_country']}"
    elif code == "SHARED_NEW_BENEFICIARY":
        if f["new_recipient"] and f["recipient_senders_24h"] >= p["min_other_senders_24h"] and recent_beneficiary:
            return f"{f['recipient_senders_24h']} other accounts paid this beneficiary in the last 24 hours"
    elif code == "REPEAT_NEW_BENEFICIARY":
        prior = f.get("pair_requests_24h", 0)
        if has_history and f["new_recipient"] and prior >= p["min_prior_payments_24h"]:
            return f"{prior + 1} payments to this new beneficiary in 24 hours"
    elif code == "NEW_BENEFICIARY_HIGH_AMOUNT":
        if has_history and f["new_recipient"] and z is not None and z >= p["min_zscore"]:
            return f"z-score {z} to a beneficiary without an established payment history"
    elif code == "NEW_BENEFICIARY":
        if has_history and f["new_recipient"]:
            hours = f.get("recipient_relationship_hours")
            return "first payment to this beneficiary" if hours is None else f"first paid {hours:.1f} hours ago"
    elif code == "PASS_THROUGH_FUNDS":
        incoming = f.get("incoming_24h", 0.0)
        outflow = f.get("sum_24h", 0.0) + f["amount"]
        if incoming >= p["min_incoming_24h"] and outflow >= p["min_outflow_share"] * incoming and recent_beneficiary:
            return f"{_money(outflow)} sent in 24 hours after receiving {_money(incoming)}"
    elif code == "NEAR_TX_CAP":
        ratio = f.get("amount_to_tx_cap_ratio") or 0.0
        if ratio >= p["min_cap_ratio"]:
            return f"{_money(f['amount'])} is {ratio:.0%} of the {_money(f.get('tx_cap'))} per-transaction limit"
    elif code == "DAILY_CAP_PRESSURE":
        ratio = f.get("daily_cumulative_to_cap_ratio") or 0.0
        if ratio >= p["min_daily_cap_ratio"] and f["new_recipient"]:
            return (f"{_money(f.get('sent_today_before', 0.0) + f['amount'])} sent today including this payment, "
                    f"{ratio:.0%} of the {_money(f.get('daily_cap'))} daily limit")
    elif code == "CAP_STRUCTURING":
        ratio = f.get("amount_to_tx_cap_ratio") or 0.0
        prior = f.get("near_cap_count_24h", 0)
        if ratio >= p["min_cap_ratio"] and prior >= p["min_prior_near_cap_24h"]:
            return f"{prior + 1} payments within 24 hours just under the {_money(f.get('tx_cap'))} per-transaction limit"
    elif code == "RECENT_ALIAS_REBOUND":
        hours = f.get("alias_rebound_age_hours")
        if hours is not None and hours <= p["max_hours"]:
            return f"phone alias rebound to a new SIM or device {hours:.1f} hours ago"
    elif code == "UNKNOWN_RTP_REQUESTER":
        if f.get("is_request_to_pay") and f["new_recipient"] and f.get("recipient_new_global"):
            return "Request-to-Pay approved for a requester nobody had an established payment history with"
    elif code == "QR_MERCHANT_MISMATCH":
        if f.get("qr_merchant_mismatch"):
            return f"the QR code names merchant {f.get('qr_payload_merchant_id')}, but the money goes to another account"
    return None


def evaluate_rules(facts: Mapping[str, Any], history_status: str, policy: Policy) -> RuleResult:
    """Fire every rule whose condition holds. Without history, history-based rules stay silent."""
    has_history = history_status != "none"
    fired: set[str] = set()
    triggered = []
    for rule in policy.rules:
        detail = _rule_check(rule, facts, has_history, fired)
        if detail is not None:
            fired.add(rule.code)
            triggered.append(TriggeredRule(rule.code, rule.family, rule.weight, rule.corroborating, rule.description,
                                           detail))
    families = tuple(sorted({t.family for t in triggered if t.corroborating}))
    return RuleResult(min(1.0, sum(t.weight for t in triggered)), tuple(triggered), families)


# ---------------------------------------------------------------------------
# Risk score and decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBreakdown:
    total: float  # unrounded; decisions use this value
    contributions: Mapping[str, float]  # 100 x weight x component, sums to `total`


def combine_scores(components: Mapping[str, float], weights: Mapping[str, float]) -> RiskBreakdown:
    contributions = {name: 100.0 * weights[name] * components[name] for name in COMPONENTS}
    return RiskBreakdown(sum(contributions.values()), contributions)


def assessment_status(history_status: str, amount: float, auth_method: str, policy: Policy) -> tuple[str, list[str]]:
    """Can the payment be judged on behaviour? Senders without history get a safeguard instead."""
    if history_status == "none":
        if amount > policy.cold_start_max_amount or auth_method not in policy.strong_auth_methods:
            return "insufficient_history", [
                ("The sender has no earlier payments, so behavioural comparisons are unavailable; payments above "
                f"{_money(policy.cold_start_max_amount)} or without strong authentication need verification.")]
        return "limited_history", ["No earlier payments; assessed with strong authentication and a small amount."]
    if history_status == "limited":
        return "limited_history", ["Fewer than five earlier payments; behavioural baselines are less reliable."]
    return "assessed", []


@dataclass(frozen=True)
class PolicyResult:
    decision: str  # Permit | Verify | Block
    decision_source: str  # confirmed_fraud_evidence | assessment_safeguard | risk_policy
    label: str
    assessment_status: str
    evidence_ids: tuple[str, ...]
    corroborating_signals: tuple[str, ...]
    explanation: str


def decide_action(risk: RiskBreakdown, rules: RuleResult, evidence: Sequence[SecurityEvidence], status: str,
                  point: OperatingPoint, policy: Policy, segment: str = "personal") -> PolicyResult:
    """The decision table, highest priority first."""
    verify, block = point.verify, point.block
    families = rules.corroborating_families
    if evidence:
        kinds = ", ".join(sorted({e.evidence_type.replace("_", " ") for e in evidence}))
        return PolicyResult(
            "Block", "confirmed_fraud_evidence", "Confirmed fraud evidence", status,
            tuple(e.evidence_id for e in evidence), families,
            f"Blocked automatically: active trusted security evidence ({kinds}) implicates this request. "
            "The anomaly score is shown for context and did not decide this action.")
    if status == "insufficient_history":
        return PolicyResult(
            "Verify", "assessment_safeguard", "Needs verification", status, (), families,
            "Sent to verification because the request cannot be assessed safely with the information available.")
    if risk.total >= block and len(families) >= policy.min_families:
        return PolicyResult(
            "Block", "risk_policy", "Preventive risk block", status, (), families,
            f"Preventive block: risk {risk.total:.1f} reached the {segment} block threshold {block:g} and "
            f"{len(families)} distinct behavioural signals corroborate it. This is a risk decision, not confirmed fraud.")
    if risk.total >= verify:
        reason = ("very high anomaly score without enough independent corroborating signals for a block"
                  if risk.total >= block else "unusual behaviour above the verification threshold")
        return PolicyResult(
            "Verify", "risk_policy", "Needs verification", status, (), families,
            f"Needs verification: {reason} (risk {risk.total:.1f}, {segment} threshold {verify:g}).")
    return PolicyResult(
        "Permit", "risk_policy", "Permitted", status, (), families,
        f"Permitted: risk {risk.total:.1f} is below the {segment} verification threshold {verify:g}.")


def decision_codes(risk: np.ndarray, families: np.ndarray, unsafe: np.ndarray, verify: float, block: float,
                   min_families: int) -> np.ndarray:
    """`decide_action` for many payments at once without evidence: 0 Permit, 1 Verify, 2 Block.

    The threshold search tries thousands of (verify, block) pairs, so it needs the array form.
    """
    blocked = ~unsafe & (risk >= block) & (families >= min_families)
    return np.where(blocked, 2, np.where(unsafe | (risk >= verify), 1, 0))
