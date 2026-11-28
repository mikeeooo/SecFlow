"""API contract pin: the JSON shape of every endpoint the frontend calls.

    uv run python scripts/api_contract.py            # record scripts/api_contract.baseline.json
    uv run python scripts/api_contract.py --check    # compare the running code with the baseline

The script boots the FastAPI app in process (TestClient, stream autostart off), calls the routes in `frontend/src/api/client.ts` with the parameters the pages send, and
records each response's shape: keys and value types, recursively, with the union shape of a
list's items. Values are ignored, so retraining or a different random sample does not matter.

`--check` fails on a missing key or a type change. Added keys are fine. `int` and `float` both
count as `number`. A field that was never `null` in the baseline must not become `null`. A field
seen only as `null` (like an empty list) has no known type, so it is not compared.

`REMOVED` lists report fields that described machinery deleted on purpose. The frontend renders
each of them conditionally or not at all. `--check` prints them and fails on any other missing key.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("api_contract.baseline.json")
sys.path.insert(0, str(ROOT))
os.environ["SECFLOW_AUTOSTART_STREAM"] = "false"
# Fields that are null by design (`ml/features.py` returns None for a never-paid beneficiary). The
# baseline sample happened to see only numbers, so a null here is not a contract change.
NULLABLE_LEAVES = {"recipient_history_days"}

REMOVED = {
    "algorithms.training.calibration_component_average_precision": "validation is no longer split into two parts",
    "algorithms.training.search.lof[].training_lof_*": "LOF training diagnostics went with the LOF grid search",
    "algorithms.training.search.*[].fit_seconds": "wall-clock timings are kept out of training_report.json",
    "algorithms.training.segment_thresholds.*.chosen": "one threshold pair serves both segments",
    "algorithms.training.segment_thresholds.*.feasible_candidates": "one threshold pair serves both segments",
    "evaluation.drift_billers": "drift subset removed",
    "evaluation.holdout_accounts": "holdout accounts removed",
    "evaluation.per_fraud_type.*.*_victim_recall": "single/multi-victim campaign split removed",
    "evaluation.per_scenario": "repeated per_fraud_type and per_anomaly_type",
    "evaluation.splits.validation_*": "validation is no longer split into two parts",
    "evaluation.unseen_scenario.campaigns": "single/multi-victim campaign split removed",
    "evaluation.unseen_scenario.*_victim": "single/multi-victim campaign split removed",
    "evaluation.unseen_scenario.per_type": "repeated per_fraud_type",
}


def kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def merge(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Union of two shapes (used for list items and repeated calls)."""
    if a is None or b is None:
        return a or b
    out: dict[str, Any] = {"types": sorted(set(a["types"]) | set(b["types"]))}
    if "keys" in a or "keys" in b:
        keys_a, keys_b = a.get("keys", {}), b.get("keys", {})
        out["keys"] = {k: merge(keys_a.get(k), keys_b.get(k)) for k in sorted(set(keys_a) | set(keys_b))}
    if "items" in a or "items" in b:
        out["items"] = merge(a.get("items"), b.get("items"))
    return out


def shape(value: Any) -> dict[str, Any]:
    t = kind(value)
    out: dict[str, Any] = {"types": [t]}
    if t == "object":
        out["keys"] = {k: shape(v) for k, v in sorted(value.items())}
    elif t == "array":
        items = None
        for item in value:
            items = merge(items, shape(item))
        out["items"] = items
    return out


def capture() -> dict[str, Any]:
    """Call every frontend route once (mutations first, stream controls last) and record the shapes."""
    from fastapi.testclient import TestClient

    from backend.api import create_app

    shapes: dict[str, Any] = {}

    def call(name: str, method: str, path: str, expect: int, **kwargs: Any) -> Any:
        response = client.request(method, path, **kwargs)
        if response.status_code != expect:
            raise SystemExit(
                f"{name}: {method} {path} answered {response.status_code}, expected {expect}: "
                f"{response.text[:300]}"
            )
        body = response.json()
        shapes[name] = merge(shapes.get(name), shape(body))
        return body

    with TestClient(create_app()) as client:
        accounts = call("accounts", "GET", "/api/accounts", 200, params={"page": 1, "page_size": 1000})
        personal = next(a for a in accounts["items"] if a["account_type"] == "personal" and a["requests"] > 5)
        options = call(
            "simulator_options_sender",
            "GET",
            "/api/simulator/options",
            200,
            params={"sender_id": personal["account_id"]},
        )
        recipient = options["sender"]["recent_recipients"][0]["id"]
        payment = {
            "request_id": "CONTRACT-0001",
            "sender_id": personal["account_id"],
            "recipient_id": recipient,
            "amount": 250.0,
            "currency": "MDL",
            "payment_scheme": "p2p_alias",
            "auth_method": "biometric",
            "failed_login_attempts": 0,
            "ip_country": "MD",
            "channel": "mobile_app",
            "transfer_type": "instant",
        }
        call("simulation_created", "POST", "/api/simulations", 201, json=payment)
        call(
            "simulation_created",
            "POST",
            "/api/simulations",
            201,
            json={
                **payment,
                "request_id": "CONTRACT-0002",
                "recipient_id": None,
                "new_recipient_name": "Ion Rusu",
                "amount": 4_900.0,
                "auth_method": "password",
                "failed_login_attempts": 3,
                "ip_country": "NL",
            },
        )
        flagged = call("simulator_options", "GET", "/api/simulator/options", 200)["flagged_beneficiaries"]
        call(
            "simulation_created",
            "POST",
            "/api/simulations",
            201,
            json={**payment, "request_id": "CONTRACT-0005", "recipient_id": flagged[0]["id"]},
        )
        call("simulation_replay", "POST", "/api/simulations", 200, json=payment)
        call(
            "error_422",
            "POST",
            "/api/simulations",
            422,
            json={**payment, "request_id": "CONTRACT-0003", "currency": "EUR"},
        )
        call(
            "error_422",
            "POST",
            "/api/simulations",
            422,
            json={**payment, "request_id": "CONTRACT-0004", "amount": 9_000.0},
        )

        pending = call(
            "transactions_filtered",
            "GET",
            "/api/transactions",
            200,
            params={
                "decision": "Verify",
                "review_status": "pending",
                "sort": "risk_desc",
                "page": 1,
                "page_size": 25,
            },
        )
        target = pending["items"][0]["request_id"]
        call("investigation", "GET", f"/api/investigations/{target}", 200)
        call(
            "review_created",
            "POST",
            f"/api/investigations/{target}/reviews",
            201,
            json={"action": "reject", "reason": "Contract pin: confirmed with the customer", "actor": "Pin"},
        )
        second = pending["items"][1]["request_id"]
        call(
            "review_created",
            "POST",
            f"/api/investigations/{second}/reviews",
            201,
            json={"action": "approve", "reason": "Contract pin: customer confirmed the payee"},
        )
        call("error_404", "GET", "/api/investigations/NOPE-404", 404)

        call("health", "GET", "/api/health", 200)
        call("overview", "GET", "/api/overview", 200)
        call("transactions", "GET", "/api/transactions", 200, params={"page": 1, "page_size": 25})
        call(
            "transactions_filtered",
            "GET",
            "/api/transactions",
            200,
            params={
                "q": personal["display_name"].split()[0],
                "decision_source": "risk_policy",
                "origin": "baseline",
                "date_from": "2025-01-01",
                "min_score": 0,
                "max_score": 100,
                "sort": "time_asc",
                "page": 1,
                "page_size": 25,
            },
        )
        call(
            "accounts",
            "GET",
            "/api/accounts",
            200,
            params={"account_type": "business", "page": 1, "page_size": 1000},
        )
        call("account_detail", "GET", f"/api/accounts/{personal['account_id']}", 200)
        call("parties", "GET", "/api/parties", 200, params={"q": personal["display_name"][:3], "limit": 12})
        call("simulator_options", "GET", "/api/simulator/options", 200)
        call("investigation", "GET", f"/api/investigations/{target}", 200)
        call("investigation", "GET", "/api/investigations/CONTRACT-0002", 200)
        call("investigation", "GET", "/api/investigations/CONTRACT-0005", 200)
        call("reviews", "GET", "/api/reviews", 200, params={"sort": "newest", "page": 1, "page_size": 50})
        call(
            "reviews",
            "GET",
            "/api/reviews",
            200,
            params={"action": "reject", "q": "pin", "sort": "oldest", "page": 1, "page_size": 50},
        )
        call("live_transfers", "GET", "/api/live-transfers", 200, params={"after": 0, "limit": 100})
        call("algorithms", "GET", "/api/algorithms", 200)
        call("evaluation", "GET", "/api/evaluation", 200)
        call("dashboard_history", "GET", "/api/dashboard", 200, params={"window": "history"})
        call("dashboard_live", "GET", "/api/dashboard", 200, params={"window": "live"})
        call("cluster_graph", "GET", "/api/algorithms/cluster-graph", 200, params={"anomalies": 150})

        call("stream_status", "GET", "/api/simulation-stream", 200)
        call("stream_status", "POST", "/api/simulation-stream/mode", 200, json={"mode": "batch_5"})
        call("stream_status", "POST", "/api/simulation-stream/start", 200)
        call("stream_status", "POST", "/api/simulation-stream/stop", 200)
    return shapes


def compare(base: dict[str, Any] | None, new: dict[str, Any] | None, path: str, problems: list[str]) -> None:
    if base is None:  # nothing was observed in the baseline (an empty list), so nothing to hold
        return
    if new is None:
        problems.append(f"missing: {path}")
        return
    base_types, new_types = set(base["types"]), set(new["types"])
    if path.rsplit(".", 1)[-1] in NULLABLE_LEAVES:
        base_types |= {"null"}
    if base_types != {"null"} and not new_types <= base_types:
        problems.append(f"type change: {path} was {sorted(base_types)}, now {sorted(new_types)}")
    for key, child in base.get("keys", {}).items():
        if key not in new.get("keys", {}):
            problems.append(f"missing: {path}.{key}")
        else:
            compare(child, new["keys"][key], f"{path}.{key}", problems)
    if base.get("items") is not None:
        compare(base["items"], new.get("items"), f"{path}[]", problems)


def check(shapes: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(problems, removed): differences from the baseline, minus the paths in `REMOVED`."""
    found: list[str] = []
    for name, base in json.loads(BASELINE.read_text()).items():
        compare(base, shapes.get(name), name, found)
    removed = [
        p
        for p in found
        if p.startswith("missing: ")
        and any(fnmatch.fnmatchcase(p.removeprefix("missing: "), pattern) for pattern in REMOVED)
    ]
    return [p for p in found if p not in removed], removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="compare with the committed baseline")
    args = parser.parse_args()
    logging.disable(logging.INFO)
    shapes = capture()
    if not args.check:
        BASELINE.write_text(json.dumps(shapes, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(shapes)} response shapes -> {BASELINE.relative_to(ROOT)}")
        return 0
    problems, removed = check(shapes)
    for pattern, reason in REMOVED.items():
        if hits := sum(fnmatch.fnmatchcase(p.removeprefix("missing: "), pattern) for p in removed):
            print(f"removed on purpose: {pattern} ({hits} path(s)): {reason}")
    for problem in problems:
        print(problem)
    print(
        f"{len(shapes)} responses checked: {sum(p.startswith('missing') for p in problems)} missing key(s), "
        f"{sum(p.startswith('type') for p in problems)} type change(s), {len(removed)} path(s) removed on purpose"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
