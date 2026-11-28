# SecFlow Phase 1 audit

This audit measures where SecFlow stands against a professional standard for a university project. The standard is reproducible, verifiable and clean, not production-scaled. Every verdict below comes from a tool run on commit `04d53d4` (`origin/main` on 2026-11-28). The audit changes no existing code, config, data or docs.

- Raw tool output is in `docs/audit/raw/<tool>.txt`. Citations use `file:line` in those files. Local absolute paths in them are replaced with `<repo>`, `<main-clone>` and `<tmp>`.
- The ML numbers come from `scripts/experiments.py`, which writes `docs/audit/ml_experiments.json`. Rerun it with `uv run python scripts/experiments.py`. Two runs gave byte-identical JSON.
- Machine: macOS on Apple silicon, 8 cores, load average 2.3 to 2.8 during the runs. Python 3.14.3 and Node 24.8.0.

## Verdict against the done criteria

| Criterion | Verdict | Evidence |
|---|---|---|
| Green CI on a fresh clone | NOT VERIFIED | The repo has no CI workflow (`hygiene.txt:81`). On a fresh clone every check in the README passes (`fresh-clone.txt:6-73`). Two of the proposed gates fail today: `ruff format --check` on 26 files (`ruff.txt:35`) and `mypy --strict` with 105 errors (`mypy.txt:115`). The frontend has no ESLint config (`eslint.txt:1-4`). |
| Zero gitleaks and pip-audit findings | VERIFIED | gitleaks found no leaks in all 72 non-merge commits on every ref and in the working tree (`gitleaks.txt:8,13`, `hygiene.txt:90-91`). pip-audit found no known vulnerabilities in the 37 locked packages (`pip-audit.txt:2,9`). npm audit found 0 (`npm-audit.txt:2,6`). |
| Passing import contracts | VERIFIED | `backend -> ml -> config` and `backend -> ml -> datagen -> config` are both kept. A reversed sanity contract breaks as expected, so the linter does read the graph (`import-linter.txt:80-87`). |
| Metrics reproducible from a seed | VERIFIED | `generate` rebuilt all three data files and the manifest byte for byte. Two `train` runs gave the committed model bundle, metadata and operating point byte for byte. `evaluate` reproduced `artifacts/evaluation.json` except `generated_at` and latency (`reproducibility.txt:11-52`). Only this machine was tested. |
| Metrics beat baselines with non-overlapping CIs | NOT VERIFIED | The combined score does not beat rules alone. Test AP is 0.668 [0.621, 0.714] for the combined score and 0.675 [0.628, 0.721] for rules alone. The paired difference is -0.007 [-0.023, 0.009] (`experiments.txt:5-6,15`). |
| Shuffled-label control at chance | VERIFIED | Shuffled test labels give AP 0.0245 on average over 200 shuffles, range 0.0210 to 0.0288. The fraud prevalence is 0.0240 (`experiments.txt:33`). No such control exists in the code or tests. A grep for `shuffl` and `permut` finds only `datagen/generator.py:319`. |

## Findings

The table is ranked by severity, then by cost to fix. Sizes are estimated changed lines, excluding lockfiles and generated reports.

| Id | Finding | Severity | Cost | Evidence | Proposed fix | Size |
|---|---|---|---|---|---|---:|
| F01 | The combined risk score does not beat its best single component. Rules alone match or beat it on test AP and on recall at 5% FPR. In the ablation, dropping IsolationForest or LOF costs no AP that the bootstrap can separate from zero. | high | S to report, M to act | `experiments.txt:5-6,15-17,26-29` | State the result in `docs/ALGORITHMS.md` and the README. The operator decides which metric is good enough and whether to change the weights. | 40 |
| F03 | No test encodes the controls. Nothing asserts that shuffled labels score at chance or compares the combined score with rules alone. | high | S | `experiments.txt:33`, grep result above | Add a test for the shuffled-label control and a report test for the baseline gap. Run `scripts/experiments.py` in CI. | 80 |
| F02 | The repo has no CI. Nothing runs the checks on a pull request. | high | M | `hygiene.txt:81` | Add `.github/workflows/ci.yml` that runs every Phase 1 check as a gate. | 180 |
| F04 | `ruff format --check` fails. 26 of 41 Python files would change. | medium | S | `ruff.txt:9-35,39` | Apply `ruff format` in one generated commit and list it in `.git-blame-ignore-revs`. | 3,713 generated |
| F06 | Settings are parsed by hand from 7 `SECFLOW_*` variables. A typo in `SECFLOW_STREAM_INTERVAL` raises a bare `ValueError` at import. Only `SECFLOW_AUTOSTART_STREAM` is documented. The `.env*` ignore rule also ignores `.env.example`, so a template cannot be committed. | medium | S | `config/settings.py:44-52`, `docs/ARCHITECTURE.md:94`, `hygiene.txt:4` | Use `pydantic-settings`. Add `.env.example` and a `!.env.example` line in `.gitignore`. | 90 |
| F08 | Seven docs claims disagree with the measured numbers or the code. See [Docs claims to correct](#docs-claims-to-correct). | medium | S | `docs-claims.txt` | Correct them in Phase 3. | 25 |
| F05 | `mypy --strict` reports 105 errors in 21 files. 58 are outside tests and missing stubs. Most are `str` values passed where the response schemas expect `Literal` types (`backend/views.py`, `backend/service.py`), and `float \| None` values in the threshold search rows (`ml/training.py:138-155`). None of the ones read is a runtime bug. | medium | M | `mypy.txt:115`, `typecheck_summary.txt:1-36` | Add `[tool.mypy]` with `strict = true`, stub overrides for sklearn and joblib, and relaxed rules for tests. Type the threshold rows as a `TypedDict`. Fix in two PRs. | 300 |
| F07 | `storage/train.json` is 79.8 MB. GitHub warns above 50 MB and rejects files above 100 MB. History also holds older data blobs of 20 to 32 MB and a 29.9 MB older model bundle. A fresh clone downloads 81 MB of `.git`. | medium | S to keep, L to rewrite history | `hygiene.txt:24,55-66`, `fresh-clone.txt:77-78` | Operator decision. `generate` rebuilds the data byte for byte in 7.5 s, so the file could be generated instead of tracked. Rewriting history needs a force-push. | 0 to 20 |
| F09 | `training_report.json` stores wall-clock timings, so every retrain changes a tracked file. `evaluation.json` stores `generated_at` and latency, so a regenerate-and-diff check needs a key filter. | low | S | `reproducibility.txt:26-32,37-52` | Move timings out of the tracked reports or round them away. | 20 |
| F11 | The SPA fallback answers every unknown non-API path with `index.html` and status 200, `/robots.txt` included. Lighthouse SEO is 91 because of this one audit. | low | S | `backend/api.py:229-236`, `lighthouse.txt:4,16` | Add `frontend/public/robots.txt`. Return 404 for unknown paths that have a file extension. | 10 |
| F10 | The frontend has no ESLint config. A minimal config finds 1 warning, `react-hooks/exhaustive-deps` at `frontend/src/pages/Simulator.tsx:136`. It is a false positive: `nearCap` reads only `median` and `maxAmount`, which are already dependencies. | low | S | `eslint.txt:1-34` | Add `frontend/eslint.config.js` and an `npm run lint` script. | 40 |
| F12 | `backend/cli.py` has 0% line coverage. Coverage tooling is not in the dev dependencies and no coverage floor exists. | low | S | `pytest.txt:18,48-58` | Add `pytest-cov`. Add a CLI smoke test that runs `generate`, `train` and `evaluate` into a temp dir through the `SECFLOW_*` variables. | 50 |
| F13 | `requires-python` says 3.12 or later, but `.python-version` pins 3.14. Tests pass on 3.13 and 3.14. Python 3.12 was not tested because no 3.12 interpreter is installed here. | low | S | `hygiene.txt:77-79`, `pytest-py313.txt:13` | Test 3.12 and 3.14 in a CI matrix, or raise the floor. | 5 |
| F14 | `ml.evaluation` imports `datagen.generator`, the 1,636-line generator, for two constants. | low | S | `import-graph.txt:62-69`, `ml/evaluation.py:23` | Move `ALL_FRAUD_TYPES` and `TEST_ONLY_FRAUD_TYPES` to `datagen/records.py`. | 10 |
| F16 | The test run warns that Starlette's `TestClient` with `httpx` is deprecated. | low | S | `pytest.txt:6` | Follow the warning when the dependency is bumped. | 5 |
| F18 | All intervals are payment-level, the Wilson intervals in `ml/evaluation.py` and the bootstrap here. Fraud arrives in campaigns, so the intervals are too narrow. | low | S | `ml/evaluation.py:39`, `ml_experiments.json` `method.bootstrap` | Add a campaign-level bootstrap to `scripts/experiments.py`. Every test row has a `campaign_id`. | 40 |
| F17 | The repo has no LICENSE, no ADRs and no `.env.example`. | low | S | `hygiene.txt:82-86` | Add ADRs in `docs/adr/`. The license is an operator decision. | 150 |
| F15 | Two IsolationForest feature groups carry no signal on test. Permuting the velocity group raises AP by 0.0138 on average, and all 20 permutations raise it. Permuting device and sign-in changes AP by -0.0010 to +0.0087. | low | M | `experiments.txt:51-60` | Drop the groups from the IsolationForest input only with a retrain and a new frozen evaluation. The test set was already seen, so this is an operator decision. | 30 |

## ML experiments

`scripts/experiments.py` scores validation and test through `ml.assess.Assessor`, the same path as the app and `ml/evaluation.py`. It reads the committed model bundle and `storage/*.json` and retrains nothing. It checks three facts before it reports:

- Its fast average precision equals sklearn's, with and without bootstrap weights.
- The Assessor's risk equals `100 x components @ weights`.
- `select_thresholds` on these validation scores gives the committed operating point, Verify 68 and Block 86.

The component average precisions match `artifacts/evaluation.json` to three decimals.

Method notes:

- AP is sklearn's average precision, the PR-AUC.
- Recall at a fixed FPR uses the lowest cut that keeps at most that share of validation legitimate payments strictly above it. The same cut is applied to test.
- Intervals are percentile intervals from 1,000 payment-level bootstrap resamples of test, seed 0. F18 explains why they are too narrow.
- The best single component is the one with the highest validation AP, so test labels do not choose it.

### Combined score against each component alone

| Score | Validation AP | Test AP [95% CI] | Recall @ 1% FPR [95% CI] | Test FPR | Recall @ 5% FPR [95% CI] | Test FPR |
|---|---:|---:|---:|---:|---:|---:|
| combined_fixed_weights | 0.719 | 0.668 [0.621, 0.714] | 0.669 [0.620, 0.718] | 0.0129 | 0.836 [0.797, 0.874] | 0.0675 |
| rules_only | 0.696 | 0.675 [0.628, 0.721] | 0.661 [0.610, 0.710] | 0.0109 | 0.861 [0.825, 0.898] | 0.0411 |
| isolation_forest_only | 0.566 | 0.495 [0.445, 0.550] | 0.522 [0.475, 0.573] | 0.0180 | 0.681 [0.632, 0.727] | 0.0676 |
| lof_only | 0.056 | 0.051 [0.036, 0.072] | 0.061 [0.038, 0.086] | 0.0202 | 0.103 [0.071, 0.136] | 0.0568 |
| dbscan_only | 0.028 | 0.035 [0.028, 0.044] | 0.000 [0.000, 0.000] | 0.0000 | 0.139 [0.101, 0.175] | 0.0408 |

| Metric | Combined 95% CI | Rules-only 95% CI | Non-overlapping, combined higher | Paired difference [95% CI] |
|---|---:|---:|---|---:|
| AP | [0.621, 0.714] | [0.628, 0.721] | no | -0.007 [-0.023, 0.009] |
| Recall @ 1% FPR | [0.620, 0.718] | [0.610, 0.710] | no | +0.008 [-0.025, 0.040] |
| Recall @ 5% FPR | [0.797, 0.874] | [0.825, 0.898] | no | -0.026 [-0.055, 0.003] |

The combination does not beat the best single component. On validation the combined score leads, 0.719 against 0.696. On test the order reverses. At the 5% target, rules alone flag more fraud while they flag fewer legitimate payments, 4.1% against 6.8%, so the rules lead does not come from flagging more. For the combined score, the cut set at 5% validation FPR gives 6.75% FPR on test. `docs/DATASET.md:173` documents a 12% rise in routine amounts from day 86. That drift is a possible cause, and this audit did not isolate it.

The rules are not label-free. `docs/ALGORITHMS.md:105-115` says five rule weights were lowered on validation counts, after a test result prompted the search.

### Leave-one-out ablation

The fixed weights in `config/policy.json` are IF 0.65, LOF 0.05, DBSCAN 0.00 and rules 0.30. A full simplex search on validation (step 0.05, 1,771 points) picks IF 0.40, LOF 0.05, DBSCAN 0.00 and rules 0.55. It has validation AP 0.725 against 0.719 for the fixed weights, and test AP 0.689 [0.643, 0.734] against 0.668. Drops are measured against this re-searched four-component score.

| Dropped | Re-searched weights | Test AP | AP drop [paired 95% CI] | Recall @ 5% FPR | Recall drop [paired 95% CI] |
|---|---|---:|---:|---:|---:|
| isolation_forest | lof=0.05, rules=0.95 | 0.686 | +0.003 [-0.006, 0.010] | 0.908 | -0.036 [-0.059, -0.017] |
| lof | isolation_forest=0.50, rules=0.50 | 0.684 | +0.005 [-0.000, 0.011] | 0.858 | +0.014 [0.000, 0.029] |
| dbscan | isolation_forest=0.40, lof=0.05, rules=0.55 | 0.689 | +0.000 [0.000, 0.000] | 0.872 | +0.000 [0.000, 0.000] |
| rules | isolation_forest=1.00 | 0.495 | +0.194 [0.158, 0.229] | 0.681 | +0.192 [0.150, 0.235] |

Only the rules carry measurable weight. Without IsolationForest, recall at 5% FPR rises by 0.036, and that interval excludes zero. DBSCAN already has weight 0, so dropping it changes nothing.

### Shuffled-label controls

(i) Shuffled test labels. The combined score has true test AP 0.668. Over 200 shuffles of the test labels, AP averages 0.0245, range 0.0210 to 0.0288. The first shuffle gives 0.0224. The fraud prevalence is 0.0240, so the score is at chance once the labels carry no signal.

(ii) Shuffled validation labels. The weight search and the threshold search ran on shuffled validation labels, and the result was scored on the true test labels. Ten seeds:

| Seed | Weights from shuffled validation | True test AP | Recall @ 1% FPR | Recall @ 5% FPR | Policy Verify/Block | Policy recall (V or B) |
|---|---|---:|---:|---:|---|---:|
| true labels | isolation_forest=0.40, lof=0.05, rules=0.55 | 0.689 | | | 68/86 (fixed weights) | 0.781 |
| 1000 | lof=1.00 | 0.051 | 0.058 | 0.103 | none feasible | n/a |
| 1001 | lof=0.85, dbscan=0.05, rules=0.10 | 0.130 | 0.131 | 0.183 | 83/100 | 0.167 |
| 1002 | isolation_forest=0.45, lof=0.30, dbscan=0.25 | 0.155 | 0.067 | 0.367 | 69/100 | 0.322 |
| 1003 | isolation_forest=0.70, dbscan=0.05, rules=0.25 | 0.639 | 0.389 | 0.747 | none feasible | n/a |
| 1004 | lof=0.95, rules=0.05 | 0.104 | 0.097 | 0.131 | 92/100 | 0.117 |
| 1005 | isolation_forest=0.55, lof=0.05, dbscan=0.40 | 0.195 | 0.072 | 0.344 | 65/100 | 0.131 |
| 1006 | isolation_forest=0.05, lof=0.25, dbscan=0.70 | 0.073 | 0.047 | 0.192 | 43/100 | 0.119 |
| 1007 | isolation_forest=0.35, lof=0.60, dbscan=0.05 | 0.148 | 0.122 | 0.322 | 83/100 | 0.303 |
| 1008 | isolation_forest=0.60, lof=0.10, dbscan=0.30 | 0.191 | 0.067 | 0.389 | 71/100 | 0.128 |
| 1009 | lof=0.05, dbscan=0.10, rules=0.85 | 0.670 | 0.394 | 0.797 | 42/100 | 0.506 |

The median true test AP over the ten seeds is 0.152, against 0.689 with true labels. This control is not expected to reach chance. The detectors are unsupervised and the rules are hand-weighted, so any weight vector still ranks some fraud. The noise search picks a random point on the simplex. When it lands on rules or IsolationForest (seeds 1003 and 1009), test AP stays near 0.64 to 0.67. In 2 of 10 seeds no Verify and Block pair meets the goals. The control shows that the label-driven searches add the gap between 0.15 and 0.69. It does not show label leakage. The policy recall in the true-labels row, 0.781, matches the model-only recall in `artifacts/evaluation.json` (0.7806).

### IsolationForest permutation importance

Each feature group of the test payment vector was permuted jointly, 20 times per group, and the committed IsolationForest was rescored. The base test AP on raw scores is 0.497.

| Group | Features | AP drop mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| mia_specific | 7 | +0.2721 | 0.0102 | +0.2495 | +0.2923 |
| beneficiary | 8 | +0.1798 | 0.0070 | +0.1666 | +0.1985 |
| amount | 3 | +0.0617 | 0.0039 | +0.0545 | +0.0675 |
| time | 3 | +0.0138 | 0.0047 | +0.0041 | +0.0212 |
| device_auth | 4 | -0.0010 | 0.0047 | -0.0098 | +0.0087 |
| velocity | 6 | -0.0138 | 0.0067 | -0.0264 | -0.0020 |

The MIA-specific and beneficiary groups carry most of the IsolationForest signal. The velocity and device and sign-in groups could be dropped from the IsolationForest input. The rules still use them. Permutation measures the trained forest, not a retrained one, so the size of the gain after a retrain is unknown. The group mapping is `FEATURE_GROUPS` in `scripts/experiments.py`.

## Prototype results

### mypy --strict against pyright strict

| Checker | Errors | Wall time, 3 runs after a warm-up | Errors from untyped libraries |
|---|---:|---|---:|
| mypy 1.19.1 `--strict` | 105 in 21 files | 3.76 to 3.88 s | 13 `import-untyped` |
| pyright 1.1.408 strict | 687 in 20 files | 3.52 to 3.65 s | 552 `reportUnknown*` and `reportMissingTypeStubs` |

Sources: `mypy.txt:115-122`, `pyright.txt:1228-1233`, `typecheck_summary.txt`. Both checkers read the same 32 files. The first pyright run checked 0 files because its temporary config used absolute paths, which pyright ignores. The audit discarded that run and reran with relative paths.

Use mypy `--strict`. Speed does not decide it, because pyright is about 0.2 s faster. pyright's strict mode reports every value that comes from untyped sklearn and joblib calls, and those 552 errors would need casts throughout `ml/`. mypy treats those values as `Any` once the stubs are overridden, which leaves about 90 errors to fix in two PRs.

### LOF implementation

The premise is wrong. SecFlow uses sklearn's `LocalOutlierFactor(novelty=True)`, imported at `ml/model.py:25` and fitted at `ml/training.py:93`. No custom LOF exists. The custom code is the DBSCAN adapter `DbscanContext` at `ml/model.py:67-103`. It scores new profiles by their distance to the frozen training core samples, because sklearn's DBSCAN has no `predict`.

### Placing datagen in the layers

The actual import graph (`import-graph.txt:1-13`):

- `backend` imports `ml`, `datagen` and `config`.
- `ml` imports `datagen` (records, dataset, generator) and `config`.
- `datagen` imports only `config` (`datagen.dataset` and `datagen.generator` import `config.limits`).
- `config` imports no other first-party package.
- `scripts/api_contract.py` imports `backend`.

`datagen` already sits between `ml` and `config`. The contract `backend -> ml -> datagen -> config` holds today with no change (`import-linter.txt:81`). The graph has 7 `ml -> datagen` module edges, because `ml` uses `datagen.records` for its domain types. A contract that placed `datagen` above `ml` would break on all 7. Adopt `backend -> ml -> datagen -> config` as the contract in `pyproject.toml` and record it in an ADR. F14 removes the one odd edge, `ml.evaluation -> datagen.generator`.

## Other measurements

| Check | Result | Raw output |
|---|---|---|
| `ruff check .` (ruff 0.17.3 defaults, 407 rules) | passes | `ruff.txt:3-6` |
| ruff with `E,F,W,I,B,UP,SIM,RUF`, for sizing only | 313 findings, 305 of them `E501` | `ruff.txt:41-47` |
| `pytest --cov` | 62 passed in 57 s. Line coverage 92.5% `backend`, 99.0% `ml`, 98.5% `datagen`, 98.4% `config`. Branch coverage was not measured. | `pytest.txt:48-58` |
| `tsc --noEmit` | passes | `tsc.txt:3` |
| `vitest` | 26 tests in 5 files pass | `vitest.txt:11-12` |
| `vite build` | one JS chunk of 455 kB (137 kB at gzip -9) and 33 kB of CSS | `vite-build.txt:9-23` |
| Lighthouse 12.8.2, desktop preset, 3 runs | performance 99 to 100, accessibility 100, best practices 100, SEO 91 | `lighthouse.txt:3-16` |
| App decision mix on startup | Verify 5.00%, Block 0.88% of 15,000 payments | `app-decisions.txt` |
| `uv lock --check` | lockfile is current | `hygiene.txt:71-73` |
| Ignore rules | `.env.local`, `.vercel/` and `.DS_Store` are ignored and untracked. `artifacts/` is tracked and not ignored. | `hygiene.txt:1-20` |

Lighthouse ran on port 8031, not 8021. Port 8021 was already bound on this machine, and uvicorn exited with `EADDRINUSE`. The server was stopped after the runs.

`artifacts/` is tracked on purpose. A reader sees the evaluation without running anything, and `evaluate` reproduces it exactly today. The cost is that nothing stops it from going stale when the code changes. A CI step that regenerates it and diffs everything except `generated_at` and latency would catch that. Whether to keep it tracked is the operator's call.

## Docs claims to correct

The docs are accurate overall. The claims checker recomputed 43 claims from the data, the reports and the code (`docs-claims.txt`). These are the ones that do not hold:

| Where | Claim | Evidence |
|---|---|---|
| `README.md:189`, `docs/ALGORITHMS.md:239` | Latency p95 is 5.3 ms | `artifacts/evaluation.json` says p95 4.685 ms. Two reruns gave 4.665 and 4.692 ms (`reproducibility.txt:37,46`). |
| `docs/ALGORITHMS.md:9` | Every number on the page comes from `training_report.json` and `evaluation.json` | The rule-weight table at lines 105-115 and the 5.81% at line 244 appear in neither file. |
| `docs/ALGORITHMS.md:125-126`, `config/policy.json:17` | The fixed weights are the ones an exhaustive validation search picked in an earlier version | Today an exhaustive search picks IF 0.40, LOF 0.05, DBSCAN 0.00 and rules 0.55, at validation AP 0.725 against 0.719. The table at lines 132-139 already shows a better 50/50 mix. |
| `docs/ALGORITHMS.md:205-211`, `README.md:3-9` | The table shows rules at AP 0.675 above the combined score at 0.668, but the text never says so. The README presents the four-part score as the method. | `experiments.txt:5-17`. Rules alone are not beaten with or without non-overlapping intervals. |
| `docs/ALGORITHMS.md:57-58`, `ml/model.py:32` | 200 trees give stable scores. LOF works best on the amount features and gave the best validation AP. | No script or report in the repo measures either claim. The training report has one LOF configuration and one IsolationForest configuration. |
| `docs/DATASET.md:102` | `datagen.dataset.load_partition(path)` reads one file | No `load_partition` function exists. Only `load_partitions` does. |
| `docs/DESIGN.md:24` | White text on `#C4372B` has contrast 5.4:1 | The WCAG formula gives 5.35:1. |
| `README.md:33` | Python 3.12 or later is supported | Tests pass on 3.13 and 3.14. Python 3.12 was not tested. |

Claims checked and confirmed include the dataset counts, the scheme counts, the evidence counts, the score bands, the recall per fraud type, the design contrast ratios, and the "about 5% Verify in the app" note.

## Proposed fix PRs

Each PR is under about 400 changed lines, except PR 1, which is pure `ruff format` output that a reviewer verifies by rerunning the tool. The order follows the ranking, except where a later fix needs an earlier one.

1. `style: apply ruff format to the Python sources` (F04, 3,713 generated lines). Add the commit to `.git-blame-ignore-revs`.
2. `ci: gate pull requests on the Phase 1 checks` (F02, F12, F13, about 180 lines). One workflow with these jobs:
	- python, on 3.12 and 3.14: `uv sync --frozen`, `ruff check`, `ruff format --check`, `pytest --cov` with a floor, `lint-imports` with the contracts in `pyproject.toml`, `scripts/api_contract.py --check`, and `mypy --strict` as non-blocking until PR 7.
	- supply chain: `pip-audit` on `uv export --frozen`, and `npm audit`.
	- secrets: gitleaks with `fetch-depth: 0`.
	- frontend: `npm ci`, `tsc`, ESLint after PR 5, `vitest`, `vite build`.
	- reproducibility: `generate` into a temp dir and compare bytes. `train` into a temp dir and compare the bundle and metadata. `evaluate` and diff `evaluation.json` without the volatile keys.
	- lighthouse: serve the build from uvicorn and fail below 100 accessibility.
3. `test(ml): assert the shuffled-label control and report the rules-only baseline` (F03, F18, about 80 lines). The test asserts that shuffled test labels score within a band around the prevalence. The baseline assertion waits for the operator's metric decision, because it would fail today.
4. `refactor(config): load settings with pydantic-settings` (F06, about 90 lines). Includes `.env.example` and the `.gitignore` exception.
5. `build(frontend): add an ESLint flat config` (F10, about 40 lines plus the lockfile).
6. `chore(types): make config, datagen and ml pass mypy --strict` (F05 part 1, about 150 lines).
7. `chore(types): make backend pass mypy --strict` (F05 part 2, about 150 lines). This PR makes the CI mypy step blocking.
8. `fix(api): serve robots.txt instead of the SPA shell` (F11, about 10 lines).
9. `chore(ml): keep wall-clock timings out of tracked reports` (F09, about 20 lines).
10. `refactor(ml): import fraud type names from datagen.records` (F14, about 10 lines).
11. `docs(adr): record the import layers, tracked data and fixed weights` (F17, about 150 lines). Proposed ADRs in `docs/adr/`:
	- 0001, import layers `backend -> ml -> datagen -> config`.
	- 0002, tracked dataset, model bundle and evaluation report.
	- 0003, fixed weights and the rules-only result.
	- 0004, mypy as the type checker.
12. `docs: correct the measured claims` (F01, F08, about 60 lines). This is Phase 3, after the language decision.

## Operator decisions

These are not decided here:

- Monorepo or split repositories for backend and frontend.
- UI and UX direction.
- Which metric is good enough for the course. This decides what to do about F01 and whether PR 3 asserts a baseline.
- Romanian or English docs. `README.md` and `docs/TEHNIC.md` are Romanian. The other docs are English.
- Keeping the Vercel deploy.
- Keeping `storage/*.json`, `ml/models/` and `artifacts/` tracked, and whether to rewrite history to drop the old data blobs, which needs a force-push (F07).
- Whether to change the weights or the IsolationForest inputs after the test set was seen (F01, F15).
- The license (F17).

## Inconclusive

Nothing is inconclusive. Every tool installed and ran. These gaps limit the verdicts:

- Reproducibility was checked on one machine and one Python version. Cross-platform byte identity of the joblib bundle is not shown.
- Python 3.12 support was not tested.
- The bootstrap intervals are payment-level and too narrow for campaign fraud (F18).
