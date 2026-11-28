"""Rebuild the dataset, the models and the evaluation from the seed and compare them with the committed files.

    uv run python scripts/reproduce.py      # exit 1 if any committed file differs

It runs `generate`, `train` twice and `evaluate` from `backend.cli` into a temporary directory through
the SECFLOW_* variables, so the committed files are never touched.

- The dataset, the manifest, metadata.json, operating_point.json and evaluation.md must match byte for
  byte.
- training_report.json and evaluation.json must match with numbers equal to within TOLERANCE. LOF
  distances differ in the last bits between macOS and Linux, which moves LOF average precision by
  about 1e-6. No decision, threshold or reported digit changes.
- artifacts/timings.json holds wall-clock values and is not compared.
- The model bundle pickles those floats, so it is compared with a second train on the same machine,
  not with the committed bytes. The evaluation, which the rebuilt bundle produces, covers its behaviour.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NUMERIC = {"training_report.json", "evaluation.json"}  # compared within TOLERANCE, the rest by bytes
TOLERANCE = 1e-5


def differences(a: Any, b: Any, where: str = "", tolerance: float = TOLERANCE) -> list[str]:
    """Dotted paths of the leaves where two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for k in sorted(a.keys() | b.keys())
            for d in differences(a.get(k), b.get(k), f"{where}.{k}", tolerance)
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{where}[{i}]", tolerance)
        ]
    numbers = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
    same = math.isclose(a, b, rel_tol=0, abs_tol=tolerance) if numbers else a == b
    return [] if same else [f"{where}: {a!r} != {b!r}"]


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

        def cli(command: str, **overrides: Path) -> None:
            run_env = env | {name: str(path) for name, path in overrides.items()}
            subprocess.run([sys.executable, "-m", "backend.cli", command], cwd=ROOT, env=run_env, check=True)

        cli("generate")
        cli("train")
        cli("train", SECFLOW_MODELS_DIR=out / "models_again")
        cli("evaluate")

        pairs = [
            (ROOT / "storage" / name, out / "storage" / name)
            for name in ("train.json", "validation.json", "test.json")
        ]
        pairs.append((ROOT / "datagen" / "manifest.json", out / "manifest.json"))
        for name in ("metadata.json", "operating_point.json", "training_report.json"):
            pairs.append((ROOT / "ml" / "models" / name, out / "models" / name))
        pairs.append((out / "models_again" / "secflow_model.joblib", out / "models" / "secflow_model.joblib"))
        for name in ("evaluation.json", "evaluation.md"):
            pairs.append((ROOT / "artifacts" / name, out / "artifacts" / name))

        failures = 0
        for committed, rebuilt in pairs:
            if committed.name in NUMERIC:
                diff = differences(json.loads(committed.read_text()), json.loads(rebuilt.read_text()))
                how = f"json within {TOLERANCE:g}"
            else:
                diff = [] if committed.read_bytes() == rebuilt.read_bytes() else ["bytes differ"]
                how = "bytes"
            failures += bool(diff)
            name = committed.relative_to(out if committed.is_relative_to(out) else ROOT)
            print(f"{'DIFFERENT' if diff else 'same':9} {name} ({how})")
            for line in diff[:20]:
                print(f"    {line}")
    print(f"{len(pairs) - failures} of {len(pairs)} files reproduced")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
