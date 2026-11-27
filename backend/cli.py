"""SecFlow command line: generate, validate, train, evaluate.

    python -m backend.cli generate   # storage/{train,validation,test}.json + datagen/manifest.json
    python -m backend.cli validate   # structure, counts, MIA limits, partitions, hashes, fingerprints, reproducibility
    python -m backend.cli train      # fit on train.json; map (days 71-77) and select (days 78-85) on validation.json
    python -m backend.cli evaluate   # one frozen evaluation on test.json
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import load_settings
from datagen.dataset import cross_file_fingerprint_overlap, reproducibility_problems, validate_dataset
from datagen.generator import generate_dataset
from ml.policy import load_policy


def cmd_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    manifest = generate_dataset(settings.storage_dir, settings.manifest_path)
    for name, part in manifest["partitions"].items():
        print(
            f"{name}: {part['rows']:,} rows ({part['assessed_payments']:,} assessed payments, "
            f"{part['context_credits']:,} incoming credits), {part['fraud']} fraud, "
            f"{part['unusual_legitimate']} unusual-legitimate, {part['bytes'] / 1e6:.1f} MB -> {part['file']}"
        )
    print(f"manifest -> {settings.manifest_path}")
    return cmd_validate(args)


def cmd_validate(_: argparse.Namespace) -> int:
    settings = load_settings()
    problems = validate_dataset(settings.storage_dir, settings.manifest_path)
    overlap = cross_file_fingerprint_overlap(settings.storage_dir) if not problems else 0
    print(f"cross-file duplicate fingerprints: {overlap}")
    if overlap:
        problems.append("duplicate row fingerprints across files")
    if not problems:
        problems += reproducibility_problems(settings.storage_dir)
        print(f"reproducible from the seed: {'yes' if not problems else 'no'}")
    print(f"dataset: {'valid' if not problems else f'{len(problems)} problem(s)'}")
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_train(_: argparse.Namespace) -> int:
    from ml.training import train_models

    settings = load_settings()
    report = train_models(settings.train_path, settings.validation_path, settings.models_dir,
                          load_policy(settings.policy_path), settings.manifest_path)
    print(json.dumps(
        {
            "operating_point": report["operating_point"],
            "validation_component_average_precision": report["validation_component_average_precision"],
            "chosen": report["threshold_selection"]["chosen"],
            "training_seconds": report["training_seconds"],
        },
        indent=2,
    ))
    return 0


def cmd_evaluate(_: argparse.Namespace) -> int:
    from ml.evaluation import run_evaluation

    settings = load_settings()
    summary = run_evaluation(settings)
    print(json.dumps(summary["headline"], indent=2))
    print(f"Wrote {settings.artifacts_dir / 'evaluation.json'} and evaluation.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("generate", cmd_generate), ("validate", cmd_validate), ("train", cmd_train),
                          ("evaluate", cmd_evaluate)):
        commands.add_parser(name).set_defaults(handler=handler)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
