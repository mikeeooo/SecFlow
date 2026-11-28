"""SecFlow command line.

    python -m backend.cli generate   # storage/{train,validation,test}.json + datagen/manifest.json
    python -m backend.cli validate   # files, counts, MIA limits, fraud types, hashes, label stripping
    python -m backend.cli train      # fit on train.json, choose thresholds on validation.json
    python -m backend.cli evaluate   # one frozen evaluation on test.json
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import load_settings


def cmd_generate() -> int:
    from datagen.generator import generate_dataset

    settings = load_settings()
    manifest = generate_dataset(settings.storage_dir, settings.manifest_path)
    for name, part in manifest["partitions"].items():
        print(f"{name}: {part['rows']:,} rows, {part['fraud']} fraud -> {part['file']}")
    return cmd_validate()


def cmd_validate() -> int:
    from datagen.dataset import validate_dataset

    settings = load_settings()
    problems = validate_dataset(settings.storage_dir, settings.manifest_path)
    print(f"dataset: {'valid' if not problems else f'{len(problems)} problem(s)'}")
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_train() -> int:
    from ml.policy import load_policy
    from ml.training import train_models

    settings = load_settings()
    report = train_models(settings.train_path, settings.validation_path, settings.models_dir,
                          load_policy(settings.policy_path), settings.manifest_path)
    print(json.dumps({key: report[key] for key in ("operating_point", "validation_component_average_precision",
                                                     "training_seconds")}, indent=2))
    print(json.dumps({"chosen": report["threshold_selection"]["chosen"]}, indent=2))
    return 0


def cmd_evaluate() -> int:
    from ml.evaluation import run_evaluation

    settings = load_settings()
    summary = run_evaluation(settings)
    print(json.dumps(summary["headline"], indent=2))
    print(f"Wrote {settings.artifacts_dir / 'evaluation.json'} and evaluation.md")
    return 0


COMMANDS = {"generate": cmd_generate, "validate": cmd_validate, "train": cmd_train, "evaluate": cmd_evaluate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secflow", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    return COMMANDS[parser.parse_args(argv).command]()


if __name__ == "__main__":
    raise SystemExit(main())
