# SecFlow algorithm and dataset improvements — design

Date: 2026-11-26 · Status: draft for approval

## Intent

SecFlow is the year project for the course *Analiza și implementarea algoritmilor*. The deliverable
is a working product that shows how IsolationForest, LOF, DBSCAN and the rule engine are applied to
payment-risk assessment. This change improves the algorithms and the dataset so the product makes
better decisions and explains them, and replaces the current version in place.

## Goals

1. Learn the combination weights instead of using the fixed 0.35 / 0.25 / 0.10 / 0.30.
2. Keep LOF and improve it (better signal, lower cost).
3. Make the DBSCAN context score continuous instead of 0/1.
4. Add beneficiary-side features aimed at social engineering and mule recipients.
5. Use separate Verify/Block thresholds per account segment.
6. Make the dataset more realistic: an unseen fraud scenario in the test period, drift in the test
   period, an account-disjoint evaluation subset, and confidence intervals on test metrics.
7. Show the applied algorithms (learned weights, LOF subspace, DBSCAN distances, segment
   thresholds) on the Algorithm Results page; regenerate docs and numbers.

## Non-goals

- No v1/v2 comparison, ablation tables, or "before/after" views in the product. The old numbers
  remain only in git history.
- No re-implementation of scikit-learn estimators.
- No changes to persistence, scaling, multi-instance state, or the API contract beyond new
  read-only fields.

## Anti-goals (failure even if it runs)

- Legitimate false-block rate above 0.1% on test.
- Any leakage: features reading events at or after the request key; test labels used before
  freezing; labels read by the runtime service.
- Per-component contributions that no longer sum exactly to the risk total shown in the UI.

## Constraints

- One assessment path (`ml/assess.py`) stays the only route from request to decision.
- Chronological partitions stay whole days: train 1–60, calibration 61–70, validation 71–80,
  test 81–100. Demo dataset stays separate and is never used for selection or reporting.
- Bundle schema checks (`load_bundle`) must reject stale bundles; bump `FEATURE_VERSION` to
  `features-1.1.0` and generator to `secflow-generator-1.1.0`.
- Training plus evaluation should stay in the order of minutes on a laptop (today ~24 s training).
- Tests are updated to the new counts; all existing leakage, idempotency and evidence tests keep
  passing.

## Approaches considered

| Approach | Trade-off | Outcome |
|---|---|---|
| A. v2 beside v1 with ablation | Best for an analysis report; ~1.5× the work; adds comparison UI | Rejected: the product must show the applied algorithms, not version differences |
| **B. Replace in place** | Less code; old numbers only in git | **Chosen** |
| C. Algorithms first, dataset later | Lower risk; dataset gains not measurable until phase 2 | Rejected: one coherent change is simpler to present |

Combination weights:

| Option | Trade-off | Outcome |
|---|---|---|
| **1. Learned non-negative weights, same formula** | UI, contributions and explainability unchanged | **Chosen** (the course brief allows changing weights) |
| 2. Full logistic combiner | Possibly better; scores become logit-scale, UI changes | Rejected |
| 3. Keep fixed weights | Smallest gain | Rejected |

## Design

### 1. Learned weights

- Formula unchanged: `risk = 100 × Σ wᵢ · componentᵢ`, `wᵢ ≥ 0`, `Σ wᵢ = 1`, components
  `isolation_forest, lof, dbscan, rules`.
- Weights are chosen by exhaustive search on the simplex with step 0.05 (1,540 candidates with the
  LOF minimum below),
  maximising average precision on the **calibration** period (days 61–70) using its labels.
  Calibration labels were previously unused; IF/LOF CDF mappings are still fitted on calibration
  scores without labels. Thresholds stay on validation, so weights and thresholds are chosen on
  different rows.
- LOF keeps a minimum weight of 0.05 so it remains part of the decision (user requirement: LOF
  stays). Other components have minimum 0.
- Learned weights are saved in `operating_point.json` / bundle metadata. `config/policy.json`
  `weights` become the documented fallback used only when no trained weights exist.
- `combine_scores` receives the learned weights; contributions still sum exactly to the total.
  `docs/DESIGN.md` and the risk composition bar render the learned weights, not hard-coded ones.

### 2. LOF improved, not removed

- Fit on a **behaviour subspace** instead of the full vector. Candidate subspaces (fixed list,
  defined in code): amount (`log_amount`, `amount_zscore_30d`, `log_amount_to_prior_max_30d`),
  amount+velocity, amount+velocity+beneficiary, and the full vector.
- Grid: subspace × `n_neighbors` {20, 35, 50} × fit rows {20,000 seeded subsample, all 60,000}.
  Selection by validation AP; ties broken by fewer fit rows (lower memory and latency).
- Novelty rule unchanged: `score_samples` never on fitting rows.
- Bundle stores the selected feature indices; `LocalOutlierFactorDetector` slices columns.

### 3. DBSCAN continuous context

- Same frozen-core adapter. Score becomes
  `0` within `eps`, `min(1, (d − eps) / eps)` beyond `eps` (reaches 1 at `2·eps`),
  `0` with insufficient history. Status labels unchanged (`within_context`, `outside_context`,
  `insufficient_history`).
- Parameter search keeps the outside-rate constraint, now measuring separation by validation AP of
  the continuous score.

### 4. Beneficiary-side features (payment vector 20 → 24)

All computed strictly before the request key, from attempted requests and settled flows:

- `log_recipient_unique_senders_7d`: distinct senders that paid this recipient in 7 days.
- `log_recipient_inbound_24h`: amount received by the recipient in 24 hours.
- `pair_requests_24h`: this sender's payments to this recipient in 24 hours while the relationship
  is still new (<24 h); 0 for established payees.
- `recipient_new_global`: recipient first seen anywhere in the system less than 24 hours ago.

One new rule in `config/policy.json`: `REPEAT_NEW_BENEFICIARY` (family `beneficiary_risk`,
weight 0.15, non-corroborating): ≥ 2 payments to the same new beneficiary in 24 hours.

Known limit, stated in docs: social engineering that pays an established counterparty from a known
device remains hard to detect without evidence.

### 5. Segment thresholds

- Segments: `personal` and `business` (account type). Cold-start is not a segment: cold-start
  accounts exist only in the test period, so their thresholds could not be validated.
- The existing grid search runs per segment on validation with the same goals. A segment with no
  feasible point falls back to the global point.
- `OperatingPoint` gains a per-segment map; `decide_action` picks the sender's segment. The
  assessment safeguard and evidence-first priority order are unchanged.

### 6. Dataset (generator 1.1.0)

- **Unseen scenario `invoice_redirect`**, test partition (40 requests) and demo (20) only: a business
  account pays a brand-new counterparty an amount close to one of its usual supplier invoices, from
  a known device with strong sign-in. Never present in train/calibration/validation. It is meant to
  be **partly caught** (user decision): the fake supplier is new across the whole system, and in
  about half of the campaigns 2–3 businesses pay it within a day. The new beneficiary features
  should catch those; single-victim campaigns stay hard. It counts in the headline test metrics and
  is also reported on its own.
- **Drift in the test period (days 81–100)**: routine amounts ×1.12; three new popular billers
  (counterparty kind `biller`) appear on day 81, and 30% of accounts start paying one of them, with
  each account's first payment spread uniformly over days 81–100. No rule exemption for billers:
  this is the intended false-positive stress test. The demo dataset (later in time) uses the
  drifted amount level and already knows the three billers.
- **Account-disjoint subset**: 40 established accounts, chosen by a fixed seed in training (not by
  label), are excluded from every fitting/selection row (train, calibration, validation). Their
  history is still used as history. Evaluation reports metrics on their test requests separately.
- **Confidence intervals**: cluster bootstrap (1,000 resamples, seed 42) 95% CI for precision,
  recall and verify rate on test. The resampling unit is the campaign (`campaign_id`) for fraud and
  the request for legitimate traffic, because fraud arrives in correlated campaigns. Block
  precision keeps the existing Wilson interval.
- **Frozen test, honest reporting** (user decision): the test period is evaluated once after
  freezing. If an anti-goal is breached (for example false-block rate above 0.1% because of the
  drift billers), nothing is retuned; docs and the Algorithm Results page state the breach and its
  cause.
- Counts per scenario elsewhere stay as today (200 per existing scenario).

### 7. Product and docs

- Algorithm Results page: learned weights (with the formula rendered from them), LOF subspace and
  k, DBSCAN distance histogram, per-segment thresholds, test metrics with CIs, unseen-scenario and
  account-disjoint subset results.
- Regenerate `storage/data/*`, `ml/models/*`, `artifacts/evaluation.*`; update `docs/ALGORITHMS.md`,
  `docs/DATASET.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, README numbers.

## Calls made without the user (correct if wrong)

- **Dropped label noise and multi-seed runs**: they serve offline analysis and add nothing visible
  in the product; bootstrap CIs cover uncertainty instead.
- Weights learned on calibration labels (not validation) to avoid choosing weights and thresholds
  on the same rows.
- LOF minimum weight 0.05.
- Segments by account type, not cold-start.
- Drift magnitude (×1.12, three billers, 30%) and `invoice_redirect` design.
- New rule `REPEAT_NEW_BENEFICIARY` at weight 0.15; "≥ 2 payments" counts at least one earlier
  payment to the same new beneficiary in 24 h plus the current request.
- With LOF ≥ 0.05 the weight search has 1,540 candidates.
- Bootstrap unit: campaign for fraud, request for legitimate traffic.
- Drift billers: adoption spread over days 81–100, no biller exemption in rules.

## Left to the builder

- Exact code layout of the weight search and segment maps.
- Histogram bin sizes and chart choice on the Algorithm Results page.
- Test structure for new features, provided leakage tests cover each new feature.
