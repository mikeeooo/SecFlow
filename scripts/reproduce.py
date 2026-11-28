"""Rebuild the dataset, the models and the evaluation from the seed and compare them with the committed files.

    uv run python scripts/reproduce.py      # exit 1 if any committed file differs

It runs `generate`, `train` and `evaluate` from `backend.cli` into a temporary directory through the
SECFLOW_* variables, so the committed files are never touched. Data and model files must match byte for
byte. The JSON reports must match after the volatile keys in VOLATILE are removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Wall-clock values that change on every run. Paths are dotted keys; "*" matches every list item.
VOLATILE = {
    "training_report.json": [
        "training_seconds",
        "search.isolation_forest.*.fit_seconds",
        "search.lof.*.fit_seconds",
    ],
    "evaluation.json": ["generated_at", "latency_ms", "headline.latency_p95_ms"],
}


def drop(node: Any, keys: list[str]) -> None:
    head, *rest = keys
    if head == "*":
        for item in node:
            drop(item, rest)
    elif not rest:
        node.pop(head, None)
    elif head in node:
        drop(node[head], rest)


def stable_json(path: Path, volatile: list[str]) -> Any:
    data = json.loads(path.read_text())
    for key in volatile:
        drop(data, key.split("."))
    return data


def differences(a: Any, b: Any, where: str = "") -> list[str]:
    """Dotted paths of the leaves where two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d for k in sorted(a.keys() | b.keys()) for d in differences(a.get(k), b.get(k), f"{where}.{k}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in differences(x, y, f"{where}[{i}]")
        ]
    return [] if a == b else [f"{where}: {a!r} != {b!r}"]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="secflow-repro-") as tmp:
        out = Path(tmp)
        env = os.environ | {
            "SECFLOW_STORAGE_DIR": str(out / "storage"),
            "SECFLOW_MANIFEST_PATH": str(out / "manifest.json"),
            "SECFLOW_MODELS_DIR": str(out / "models"),
            "SECFLOW_ARTIFACTS_DIR": str(out / "artifacts"),
            "SECFLOW_AUTOSTART_STREAM": "false",
        }
        for command in ("generate", "train", "evaluate"):
            subprocess.run([sys.executable, "-m", "backend.cli", command], cwd=ROOT, env=env, check=True)

        pairs = [
            (ROOT / "storage" / name, out / "storage" / name)
            for name in ("train.json", "validation.json", "test.json")
        ]
        pairs.append((ROOT / "datagen" / "manifest.json", out / "manifest.json"))
        for name in ("secflow_model.joblib", "metadata.json", "operating_point.json", "training_report.json"):
            pairs.append((ROOT / "ml" / "models" / name, out / "models" / name))
        pairs.append((ROOT / "artifacts" / "evaluation.json", out / "artifacts" / "evaluation.json"))

        failures = 0
        for committed, rebuilt in pairs:
            volatile = VOLATILE.get(committed.name)
            if volatile is None:
                diff = [] if committed.read_bytes() == rebuilt.read_bytes() else ["bytes differ"]
                how = "bytes"
            else:
                diff = differences(stable_json(committed, volatile), stable_json(rebuilt, volatile))
                how = "json without " + ", ".join(volatile)
            failures += bool(diff)
            print(f"{'DIFFERENT' if diff else 'same':9} {committed.relative_to(ROOT)} ({how})")
            for line in diff[:20]:
                print(f"    {line}")
    print(f"{len(pairs) - failures} of {len(pairs)} committed files reproduced")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
