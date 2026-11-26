# Algorithm and Dataset Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SecFlow's fixed-weight pipeline in place with learned weights, an improved LOF, a continuous DBSCAN context score, beneficiary features, per-segment thresholds, and a more realistic generated dataset, then show the applied algorithms in the product.

**Architecture:** Generator 1.1.0 produces the new corpus and demo. `ml/features.py` 1.1.0 adds four beneficiary features. Training learns LOF subspace, DBSCAN parameters, combination weights (calibration labels) and per-segment thresholds (validation labels), and saves them in the bundle and `operating_point.json`. The single assessment path (`ml/assess.py`) reads weights and segment thresholds from the operating point. Evaluation stays frozen and adds cluster-bootstrap CIs and new subsets.

**Tech Stack:** Python 3.12, numpy, scikit-learn ≥ 1.7, FastAPI, pytest, React + TypeScript + Vitest.

**Spec:** `docs/superpowers/specs/2026-11-26-algorithm-dataset-improvements-design.md`

## Global Constraints

- `FEATURE_VERSION = "features-1.1.0"`; `GENERATOR_VERSION = "secflow-generator-1.1.0"`.
- Partitions unchanged: train 1–60, calibration 61–70, validation 71–80, test 81–100, whole days.
- Every feature reads only events with ordering key strictly lower than the request key.
- Risk formula: `risk = 100 × Σ wᵢ · componentᵢ`, `wᵢ ≥ 0`, `Σ wᵢ = 1`; contributions sum exactly to the total.
- Weight search step 0.05, LOF weight ≥ 0.05 (1,540 candidates), maximise calibration AP.
- Thresholds chosen on validation only; goals unchanged (verify rate ≤ 5%, false-block rate ≤ 0.1%, zero false blocks among blocks).
- Test labels opened once, after freezing. No retuning after test; breaches are reported, not fixed.
- Runtime service never reads labels, splits, or generator params.
- No scikit-learn estimator is re-implemented. No v1/v2 comparison anywhere in the product.
- Commands run through `uv run`; frontend through `npm --prefix frontend`.

## Review Focus

1. Sender account missing or of an unknown type → segment `personal`, never a crash (test in Task 6).
2. A bundle or `operating_point.json` from before this change → `BundleError`, service starts in degraded mode (test in Task 6).
3. A segment with no feasible threshold pair, or no fraud, on validation → falls back to the global pair and the report says so (test in Task 7).
4. Weight-search ties → deterministic winner (largest `rules`, then `isolation_forest`, then `dbscan`) so retraining reproduces the same weights (test in Task 5).
5. Risk score exactly at a segment threshold → same decision (`>=`) in both `decide_action` and `decide_arrays` (test in Task 6).

---

### Task 1: Generator 1.1.0 — unseen scenario and test-period drift

**Files:**
- Modify: `storage/generator.py`
- Test: `backend/tests/test_dataset.py`

**Interfaces:**
- Produces: `DatasetSpec.extra_fraud: dict[str, dict[str, int]]` (partition → scenario → requests); `DatasetSpec.drift_first_day: int | None`; `DatasetSpec.drift_amount_factor: float`; constants `DRIFT_BILLERS = 3`, `DRIFT_ADOPTION_SHARE = 0.30`; scenario label `"invoice_redirect"`; three biller counterparties `CP-00651`…`CP-00653` (kind `business`, category `biller`) appended to the base counterparties.
- Values: `TRAINING_SPEC.extra_fraud = {"test": {"invoice_redirect": 40}}`, `drift_first_day=81`, `drift_amount_factor=1.12`; `DEMO_SPEC.extra_fraud = {"demo": {"invoice_redirect": 20}}`, `drift_first_day=1`, `drift_amount_factor=1.12`. `FRAUD_SIZES["invoice_redirect"] = (1, 3)`.

- [ ] **Step 1: Update count tests and add scenario/drift tests**

In `test_training_corpus_has_exact_counts`: `requests == 100_000`, `fraud == 1_040`, `legitimate == 98_960`, `routine_legitimate == 93_960`, five existing scenarios `== 200`, `by_scenario["invoice_redirect"] == 40`, test partition `(20_000, 240, 1_000)`. Demo: `(10_000, 400, 120, 9_880)`, `by_scenario["invoice_redirect"] == 20`. Add:

```python
def test_invoice_redirect_only_in_test_partition(generated): ...
    # every invoice_redirect request has partition "test"; sender account_type == "business";
    # recipient is a counterparty first seen in the corpus on that campaign's day;
    # device is one of the sender's earlier devices; auth_method in {"biometric", "password_otp"}
def test_invoice_redirect_has_multi_victim_campaigns(generated): ...
    # at least 25% of invoice_redirect campaigns have >= 2 distinct senders
def test_drift_billers_first_paid_in_test_period(generated): ...
    # no request to CP-00651..CP-00653 before day 81; >= 80 distinct senders pay them on days 81-100;
    # first payment day per sender spans at least 15 distinct days
def test_drift_raises_routine_amounts(generated): ...
    # median routine amount on days 81-100 / median on days 61-80 between 1.05 and 1.20
def test_account_params_unchanged_by_drift():
    # _build_account_params(...) output equals a snapshot hash taken from 1.0.0
```

For the last test, compute the snapshot first from the unmodified code (`hashlib.sha256` of `json.dumps([asdict(p) for p in params], sort_keys=True)`) and hard-code it.

- [ ] **Step 2: Run tests, confirm they fail**

Run: `uv run pytest backend/tests/test_dataset.py -q`
Expected: FAIL on counts and missing scenario.

- [ ] **Step 3: Implement**

- `extra_fraud` campaigns are planned in `_plan` after the regular categories, same `CMP-` IDs.
- `_fraud_invoice_redirect(plan, day_start)`: senders are business accounts that are not cold-start; one sender if `plan.size == 1`, otherwise `plan.size` distinct business senders within 24 h; one `_new_counterparty(business_share=1.0)` shared by the campaign; amount = `_amount(sender, factor, 0.1)` with `factor` = the sender's `payee_factors` for its top business payee; known device `self.devices[sender][0]`; auth `"password_otp"` or `"biometric"`; `is_anomaly=False`.
- Drift adoption uses a **separate** RNG `np.random.default_rng(IDENTITY_SEED + 3000)` so the existing params draw sequence is unchanged; each account gets a `drift_biller` and `drift_day` (uniform in 81–100 for training, 1 for demo) with probability `DRIFT_ADOPTION_SHARE`, otherwise none. Store on `_Generator`, not in `AccountParams`.
- In `routine`: on days ≥ the account's `drift_day`, with probability 0.08 the recipient is its drift biller. On days ≥ `spec.drift_first_day`, multiply the routine amount by `drift_amount_factor`.
- Bump `GENERATOR_VERSION`; manifest records `extra_fraud`, a `drift` block, and drift biller IDs.

- [ ] **Step 4: Run tests, confirm they pass**

Run: `uv run pytest backend/tests/test_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add storage/generator.py backend/tests/test_dataset.py
git commit -m "feat(dataset): add invoice_redirect scenario and test-period drift"
```

---

### Task 2: Beneficiary features (features-1.1.0)

**Files:**
- Modify: `storage/history.py`, `ml/features.py`
- Test: `backend/tests/test_features.py`

**Interfaces:**
- Produces: `_RecipientSeries.cum_amount: list[float]` (prefix sums over request amounts to that recipient); `HistoryIndex.recipient_first_key(recipient_id) -> int | None`.
- `PAYMENT_FEATURES` appends, in this order: `"log_recipient_unique_senders_7d"`, `"log_recipient_inbound_24h"`, `"pair_requests_24h"`, `"recipient_new_global"` (24 values total).
- New facts keys: `recipient_unique_senders_7d: int`, `recipient_inbound_24h: float`, `pair_requests_24h: int`, `recipient_new_global: bool`.

- [ ] **Step 1: Write failing tests**

```python
def test_beneficiary_features_ignore_future_events(): ...
    # compute features for request R; append later requests to the same recipient;
    # recompute with R's key -> all four new values identical
def test_recipient_unique_senders_7d_excludes_events_older_than_7_days_and_self(): ...
def test_pair_requests_24h_counts_only_while_relationship_new(): ...
    # 2 earlier payments to a payee first paid 3 h ago -> 2; payee first paid 30 h ago -> 0
def test_recipient_new_global_true_until_24h_after_first_payment_by_anyone(): ...
def test_feature_vector_length_is_24():
    assert len(PAYMENT_FEATURES) == 24 and FEATURE_VERSION == "features-1.1.0"
```

- [ ] **Step 2: Run, confirm failure**

Run: `uv run pytest backend/tests/test_features.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement**

All windows use `HistoryIndex.window(keys, _key_at(t - W), key)`. Unique senders over 7 days come from `rec.senders[lo:hi]`, excluding the current sender. `recipient_new_global` = `recipient_first_key` is `None` or less than 24 h before `t`. Vector values: `log1p(unique_senders_7d)`, `log1p(inbound_24h)`, `float(pair_requests_24h)`, `1.0/0.0`.

- [ ] **Step 4: Run, confirm pass**

Run: `uv run pytest backend/tests/test_features.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add storage/history.py ml/features.py backend/tests/test_features.py
git commit -m "feat(features): add leakage-safe beneficiary features (features-1.1.0)"
```

---

### Task 3: Rule `REPEAT_NEW_BENEFICIARY`

**Files:**
- Modify: `config/policy.json`, `ml/policy.py` (`_rule_check`)
- Test: `backend/tests/test_policy.py`

**Interfaces:**
- Consumes: fact `pair_requests_24h` (Task 2).
- Produces: rule `{"code": "REPEAT_NEW_BENEFICIARY", "family": "beneficiary_risk", "weight": 0.15, "corroborating": false, "params": {"min_prior_payments_24h": 1}}`; `policy_revision` → `"1.1.0"`.

- [ ] **Step 1: Failing tests**

```python
def test_repeat_new_beneficiary_fires_on_second_payment(): # pair_requests_24h=1, new_recipient=True -> fires
def test_repeat_new_beneficiary_silent_on_first_or_established(): # pair_requests_24h=0 -> no; new_recipient=False -> no
def test_repeat_new_beneficiary_does_not_corroborate(): # its family not added to corroborating_families
```

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_policy.py -q`
- [ ] **Step 3: Implement** — fires when `has_history and f["new_recipient"] and f["pair_requests_24h"] >= p["min_prior_payments_24h"]`; detail `"{n+1} payments to this new beneficiary in 24 hours"`.
- [ ] **Step 4: Run, confirm pass** — same command.
- [ ] **Step 5: Commit** — `git commit -m "feat(policy): add REPEAT_NEW_BENEFICIARY rule"`

---

### Task 4: LOF subspace and continuous DBSCAN score

**Files:**
- Modify: `ml/algorithms/local_outlier_factor.py`, `ml/algorithms/dbscan_context.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Produces: `LocalOutlierFactorDetector(n_neighbors: int, columns: tuple[int, ...] | None = None, fit_rows: int | None = None, random_state: int = 42)`; `.params` includes `columns` and `fit_rows`; `fit` subsamples rows with the seeded RNG when `fit_rows < len(matrix)`; `raw_scores` slices the same columns.
- Produces: `DBSCANContext.score` returns `ContextResult.score = min(1.0, (d - eps) / eps)` for `d > eps`, else `0.0`; statuses unchanged.
- Produces: `LOF_SUBSPACES: dict[str, tuple[str, ...]]` in `ml/algorithms/local_outlier_factor.py`:
  - `"amount"`: `log_amount, amount_zscore_30d, log_amount_to_prior_max_30d`
  - `"amount_velocity"`: amount + `requests_1h, requests_24h, log_sum_24h, velocity_ratio_24h`
  - `"amount_velocity_beneficiary"`: amount_velocity + `new_recipient, recipient_senders_24h, log_recipient_unique_senders_7d, log_recipient_inbound_24h, pair_requests_24h, recipient_new_global`
  - `"full"`: all `PAYMENT_FEATURES`

- [ ] **Step 1: Failing tests**

```python
def test_lof_uses_only_selected_columns(): # changing a non-selected column does not change raw_scores
def test_lof_fit_rows_subsample_is_seeded(): # two fits with fit_rows=500 -> identical scores; n_samples_fit_ == 500
def test_dbscan_continuous_score():
    # within eps -> 0.0 within_context; d = 1.5*eps -> 0.5 outside_context; d >= 2*eps -> 1.0; insufficient -> 0.0
```

Update `test_dbscan_context_near_core_outside_insufficient_and_empty` to assert `score > 0` and status `outside_context` instead of `== 1.0`.

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_models.py -q`
- [ ] **Step 3: Implement** the signatures above.
- [ ] **Step 4: Run, confirm pass** — same command.
- [ ] **Step 5: Commit** — `git commit -m "feat(ml): LOF behaviour subspaces and continuous DBSCAN context score"`

---

### Task 5: Learned combination weights

**Files:**
- Create: `ml/weights.py`
- Test: `backend/tests/test_weights.py`

**Interfaces:**
- Produces: `simplex_grid(step: float = 0.05, minimums: Mapping[str, float] = {"lof": 0.05}) -> list[dict[str, float]]` over `COMPONENTS`; `search_weights(components: np.ndarray, y: np.ndarray, step: float = 0.05, minimums: Mapping[str, float] = {"lof": 0.05}) -> tuple[dict[str, float], dict[str, Any]]`, where `components` has shape `(n, 4)` in `COMPONENTS` order; report has `candidates`, `best_average_precision`, `top` (10 best rows).

- [ ] **Step 1: Failing tests**

```python
def test_simplex_grid_size_and_constraints():
    grid = simplex_grid()
    assert len(grid) == 1540
    assert all(abs(sum(w.values()) - 1) < 1e-9 and w["lof"] >= 0.05 - 1e-9 for w in grid)
def test_search_weights_prefers_informative_component(): # rules column == y, others noise -> rules weight == 0.95
def test_search_weights_tie_break_is_deterministic(): # all columns identical -> largest rules, then isolation_forest
def test_search_weights_rejects_single_class_labels(): # y all False -> ValueError
```

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_weights.py -q`
- [ ] **Step 3: Implement** — integer lattice in units of `step` (no float drift; round to 2 decimals); AP via `sklearn.metrics.average_precision_score` on `components @ w`; tie-break key `(ap, w["rules"], w["isolation_forest"], w["dbscan"])`.
- [ ] **Step 4: Run, confirm pass** — same command.
- [ ] **Step 5: Commit** — `git commit -m "feat(ml): simplex search for combination weights"`

---

### Task 6: Operating point with weights and segments; assessment path

**Files:**
- Modify: `ml/policy.py`, `ml/model.py` (`save`, `load_operating_point`), `ml/assess.py`
- Test: `backend/tests/test_policy.py`, `backend/tests/test_models.py`, `backend/tests/test_api.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class OperatingPoint:
      verify: float
      block: float
      source: str
      version: str
      weights: Mapping[str, float] | None = None  # None -> policy.weights
      segments: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # segment -> (verify, block)
      def thresholds_for(self, segment: str) -> tuple[float, float]: ...
  SEGMENTS = ("personal", "business")
  def segment_of(account: Account | None) -> str  # "business" if account_type == "business" else "personal"
  ```
- `decide_action(risk, rules, evidence, status, operating_point, policy, segment: str = "personal")`.
- `decide_arrays(risk, families, unsafe, has_evidence, verify: float | np.ndarray, block: float | np.ndarray, min_families)` accepts per-row threshold arrays.
- `finalize(request, snapshot, components, evidence_store, policy, operating_point, segment: str)`; `assess_one` / `assess_batch` compute `segment_of(account)` and pass it; combination uses `operating_point.weights or policy.weights`.
- `operating_point.json` gains `"weights"` and `"segments": {"personal": {"verify", "block"}, "business": {...}}`; `load_operating_point` raises `BundleError` when either key is missing.
- `PolicyResult.explanation` names the segment whose thresholds were used.

- [ ] **Step 1: Failing tests**

```python
def test_thresholds_for_falls_back_to_global(): # segments={} -> (verify, block)
def test_segment_of_unknown_or_missing_account_is_personal()
def test_decide_action_uses_segment_thresholds(): # business verify 70, personal 60, risk 65 -> personal Verify, business Permit
def test_decide_arrays_matches_decide_action_with_per_row_thresholds(): # random rows incl. risk exactly at threshold
def test_combination_uses_learned_weights_and_contributions_sum(): # OP weights differ from policy; total == sum(contributions)
def test_load_operating_point_rejects_file_without_weights(tmp_path): # -> BundleError
```

Change `test_api.py:224` to assert the API reports learned weights (sum 1, LOF ≥ 0.05) instead of the fixed dict.

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_policy.py backend/tests/test_models.py -q`
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run, confirm pass** — same command (tests needing retrained artifacts pass after Task 9).
- [ ] **Step 5: Commit** — `git commit -m "feat(policy): learned weights and per-segment thresholds in the operating point"`

---

### Task 7: Training integration

**Files:**
- Modify: `ml/training.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5, 6.
- Produces: `HOLDOUT_ACCOUNTS = 40`, `HOLDOUT_SEED = 4242`; `holdout_accounts(dataset: Dataset, cold_start_ids: set[str]) -> frozenset[str]` (seeded choice among non-cold-start accounts; no labels read); `select_segment_points(risk, families, unsafe, y, segments: np.ndarray, policy) -> tuple[OperatingPoint, dict]`; training report keys `weights_search` (with `"partition": "calibration"`), `segment_thresholds` (per segment: chosen row or `"fallback": true`), `holdout_accounts`, and `search.lof` rows with `subspace`, `n_neighbors`, `fit_rows`.

- [ ] **Step 1: Failing tests**

```python
def test_holdout_accounts_never_in_fit_rows(trained): # holdout disjoint from cold-start; no fit row has a holdout sender
def test_weights_learned_on_calibration_not_validation(trained): # report weights_search.partition == "calibration"
def test_lof_grid_covers_subspaces_k_and_fit_rows(trained): # 4 subspaces x 3 k x 2 fit_rows = 24 records
def test_segment_fallback_reported_when_infeasible(): # synthetic arrays, business infeasible -> global pair + fallback flag
```

Replace the existing `fit_rows == 60_000` LOF assertion with the grid assertion. `trained` is a session fixture running `train_models` on the generated corpus into `tmp_path`.

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_models.py -q`
- [ ] **Step 3: Implement**, in this order inside `train_models`:
  1. Replay keeping facts for calibration and validation rows.
  2. Fitting masks = partition mask `& ~holdout` for every partition used to fit or select.
  3. Preprocessor; IF grid (unchanged); LOF grid (subspace × {20, 35, 50} × {20_000, all rows}; ties → fewer fit rows); DBSCAN grid (select by validation AP of the continuous score, keep legitimate outside-rate ≤ 0.15).
  4. IF/LOF score mappings on calibration (no labels).
  5. `search_weights` on the calibration component matrix with calibration labels.
  6. Validation risk with learned weights; `select_segment_points` runs the existing grid per segment and falls back to the global pair when a segment has no feasible point.
  7. Save weights and segments in `operating_point.json` and metadata; model version string uses `-1.1-`.
- [ ] **Step 4: Run, confirm pass** — same command.
- [ ] **Step 5: Commit** — `git commit -m "feat(training): learned weights, LOF subspace grid, segment thresholds, holdout accounts"`

---

### Task 8: Evaluation additions

**Files:**
- Modify: `ml/evaluation.py`
- Test: `backend/tests/test_evaluation.py`

**Interfaces:**
- Produces: `cluster_bootstrap(codes: np.ndarray, y: np.ndarray, clusters: np.ndarray, n: int = 1000, seed: int = 42) -> dict[str, list[float]]` with keys `precision_verify_or_block`, `recall_verify_or_block`, `verify_rate` (95% percentile intervals); `clusters` = `campaign_id` for fraud rows, `request_id` for legitimate rows.
- Summary keys added: `confidence_intervals`, `holdout_accounts` (subset metrics), `unseen_scenario` (`invoice_redirect`: recall overall, single-victim, multi-victim), `drift_billers` (verify and block rate on requests to drift billers), `segment_thresholds`, `weights`.
- Score band edges built from the frozen thresholds: `(0, 10, 20, 30, 40, 50, verify, 70, 80, 90, block, 100)`, deduplicated and sorted (replaces the hard-coded `SCORE_BAND_EDGES`).
- DBSCAN component flag: `score > 0` (`outside_context`).

- [ ] **Step 1: Failing tests**

```python
def test_cluster_bootstrap_resamples_campaigns_as_units(): # two 5-row campaigns, one caught -> recall CI values only in {0, 0.5, 1}
def test_cluster_bootstrap_is_seeded()
def test_score_band_edges_follow_operating_point(): # verify 58, block 95 -> both are edges
def test_evaluation_report_has_new_sections(evaluated): # keys present; holdout subset requests > 0
```

- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_evaluation.py -q`
- [ ] **Step 3: Implement**; `render_markdown` prints the new sections and states any goal breach plainly.
- [ ] **Step 4: Run, confirm pass** — same command.
- [ ] **Step 5: Commit** — `git commit -m "feat(evaluation): cluster bootstrap CIs, holdout, unseen scenario and drift subsets"`

---

### Task 9: Regenerate data, retrain, evaluate

**Files:**
- Regenerate: `storage/data/training/*`, `storage/data/demo/*`, `ml/models/*`, `artifacts/evaluation.*`

- [ ] **Step 1:** `uv run python -m backend.cli generate` — Expected: manifests show generator 1.1.0 and Task 1 counts.
- [ ] **Step 2:** `uv run python -m backend.cli validate` — Expected: no errors.
- [ ] **Step 3:** `uv run python -m backend.cli train` — Expected: learned weights (sum 1, LOF ≥ 0.05) and two segment points in the report.
- [ ] **Step 4:** `uv run python -m backend.cli evaluate` — run **once**; record results as they are; do not retune.
- [ ] **Step 5:** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 6: Commit** — `git commit -m "chore: regenerate dataset 1.1.0 and retrain models"`

---

### Task 10: API and Algorithm Results page

**Files:**
- Modify: `backend/service.py` (`algorithms`, assessment detail), `backend/schemas.py`, `frontend/src/api/types.ts`, `frontend/src/pages/Algorithms.tsx`, `frontend/src/components/RiskComposition.tsx` and its callers
- Test: `backend/tests/test_api.py`, `frontend/src/__tests__/components.test.tsx`

**Interfaces:**
- `/api/algorithms`: `policy.weights` = learned weights with `policy.weights_source: "learned" | "policy_fallback"` (fallback only in degraded mode); `policy.segments`; `training.weights_search.top`; `training.search.lof` with subspaces; `distributions.dbscan` = histogram of continuous context scores (0–1, `HISTOGRAM_BINS` bins); `evaluation` excerpt with CIs, holdout, unseen scenario, drift billers.
- Assessment detail exposes `segment` plus the verify/block thresholds and weights used; `RiskComposition` takes weights and thresholds from the response instead of constants.

- [ ] **Step 1: Failing tests** — API: `weights_source == "learned"`, `set(segments) == {"personal", "business"}`, `len(distributions["dbscan"]) == HISTOGRAM_BINS`. Frontend: `RiskComposition` renders the passed weights (e.g. `0.45 ×`) and the passed segment thresholds.
- [ ] **Step 2: Run, confirm failure** — `uv run pytest backend/tests/test_api.py -q` and `npm --prefix frontend test`
- [ ] **Step 3: Implement** the Algorithm Results sections: formula from learned weights; weight-search top-10 table; LOF subspace / k / fit-rows table with the selected row marked; DBSCAN continuous-score histogram; per-segment threshold stats; test metrics with CI ranges; unseen scenario, holdout and drift-biller cards. Goal breaches appear as plain text.
- [ ] **Step 4: Run, confirm pass** — both commands, plus `npm --prefix frontend run build`.
- [ ] **Step 5: Verify in browser** — start the app from `.claude/launch.json`; open Algorithm Results and one Investigation; weights and segment thresholds render; no console errors.
- [ ] **Step 6: Commit** — `git commit -m "feat(ui): show learned weights, LOF subspace, DBSCAN distances and segment thresholds"`

---

### Task 11: Docs

**Files:**
- Modify: `docs/ALGORITHMS.md`, `docs/DATASET.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, `README.md`

- [ ] **Step 1:** Rewrite numbers and descriptions only from `ml/models/training_report.json` and `artifacts/evaluation.json`: learned weights, LOF subspace, continuous DBSCAN, new features and rule, segment thresholds, calibration labels now used for weights, generator 1.1.0 counts, `invoice_redirect`, drift, holdout subset, cluster-bootstrap CIs, any goal breach with its cause, the social-engineering limit.
- [ ] **Step 2:** `grep -rn "0\.35\|features-1\.0\.0\|generator-1\.0\.0\|1,000 / 99,000" docs README.md` — Expected: no stale hits.
- [ ] **Step 3: Commit** — `git commit -m "docs: describe improved algorithms and dataset 1.1.0"`
