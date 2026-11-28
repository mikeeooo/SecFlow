# 0002. Track the dataset, the model bundle and the evaluation report

Status: accepted, 2026-11-28. The operator has not decided whether to keep this. See "Open question".

## Context

The repo tracks the generated dataset in `storage/`, the trained bundle and its reports in `ml/models/`, and the frozen evaluation in `artifacts/`. `storage/train.json` is 79.8 MB. GitHub warns above 50 MB and rejects files above 100 MB. A fresh clone downloads 81 MB of `.git`, part of it older data blobs.

Every one of these files can be rebuilt from the seed:

- `generate` rebuilds the dataset and the manifest byte for byte in about 8 s.
- `train` rebuilds `metadata.json` and `operating_point.json` byte for byte. On the same machine it also rebuilds the bundle byte for byte.
- `evaluate` rebuilds `evaluation.json` and `evaluation.md`.

## Decision

Keep the files tracked. Then a reader sees the data, the model and the evaluation without running anything, and the Vercel deploy serves them as they are.

CI checks that the tracked files have not gone stale. `scripts/reproduce.py` rebuilds everything into a temp dir and compares the result with the committed files:

- The dataset, the manifest, `metadata.json`, `operating_point.json` and `evaluation.md` are compared byte for byte.
- `training_report.json` and `evaluation.json` are compared with numbers equal to within 1e-5. On Linux, LOF distances differ from the macOS values in the last bits, which moves LOF average precision by about 1e-6.
- The bundle is compared with a second train on the same machine, because it pickles those floats.

Wall-clock values are kept out of the compared files. Training prints its time. `evaluate` writes `generated_at` and the latency to `artifacts/timings.json`, and the app merges that file into `/api/evaluation`.

## Consequences

- A code change that changes any result fails CI until the files are regenerated and committed.
- `artifacts/timings.json` changes on every `evaluate`. It is the one tracked file that CI does not compare.
- The clone stays large.

## Open question

Generating the dataset instead of tracking it, and rewriting history to drop the old blobs, are operator decisions. A history rewrite needs a force-push.
