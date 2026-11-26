# Algorithms, policy, and measured results

All three estimators are imported from scikit-learn 1.9.1. Nothing is re-implemented. Every number
below comes from `ml/models/training_report.json` and `artifacts/evaluation.json`.

## Features

`ml/features.py` (`features-1.1.0`) builds two vectors per request, using only events before the
request.

**Payment vector (24 values):** log amount; amount z-score against the sender's 30-day amounts;
log ratio to the 30-day maximum; requests in the last hour and the last 24 hours; log 24-hour sum;
24-hour pace relative to the usual daily rate (normalised by account age when younger than 30
days); unique recipients in 24 hours; no established beneficiary relationship (never paid, or first
paid under 24 hours ago); other senders paying the beneficiary in 24 hours; log beneficiary age;
log device age; failed sign-ins in 24 hours; log hours since the previous request; log account age;
log incoming funds in 24 hours; foreign session; weak authentication; hour-of-day deviation; log
requests in 30 days; and four **beneficiary-side** values:

| Feature | Meaning |
|---|---|
| `log_recipient_unique_senders_7d` | Other senders that paid this recipient in the last 7 days |
| `log_recipient_inbound_24h` | Amount the recipient received in the last 24 hours |
| `pair_requests_24h` | This sender's earlier payments to this recipient in 24 h, while the relationship is still new |
| `recipient_new_global` | Nobody paid this recipient before, or the first payment by anyone is under 24 hours old |

**Account-context vector (8 values, for DBSCAN):** the trailing-24-hour prospective profile. It
covers request count, sum, unique and new recipients, devices, failed sign-ins, incoming funds, and
the 24-hour sum relative to the usual daily sum.

IDs, names, labels, scenarios, campaign IDs, split assignments, and generator parameters are never
numeric inputs.

## Fitting, score mapping and weight learning

| Step | Data | Labels used? |
|---|---|---|
| Winsorise (0.5/99.5%) and `StandardScaler` | train, days 1–60, minus holdout accounts (53,045) | No |
| `IsolationForest` | same rows | No |
| `LocalOutlierFactor(novelty=True)` on a behaviour subspace | seeded 20,000-row subsample of those rows | No |
| `DBSCAN` on scaled 24 h profiles | seeded 20,000-profile subsample of train profiles with ≥ 3 prior requests | No |
| IF/LOF empirical CDF mappings | calibration, days 61–70, minus holdout (8,870) | No |
| **Combination weights** | calibration, same 8,870 rows (77 fraud) | Yes |
| Parameter choice, per-segment Verify/Block thresholds | validation, days 71–80, minus holdout (8,789) | Yes |
| Final report | test, days 81–100 (20,000) | Yes, once, after freezing |

**Holdout accounts.** Forty established accounts, picked with a fixed seed and without looking at
labels, are removed from every fitting and selection row. Their history still feeds features. The
test period reports their requests as an account-disjoint subset.

**Score direction.** IF uses `-score_samples` and LOF uses `-score_samples` on unseen rows; in both,
a higher value means more anomalous. Each is mapped through the calibration-period empirical CDF
(`searchsorted(side="right") / n`), giving a value in `[0, 1]`.

**LOF, improved.** LOF stays in the index. Training searches 4 behaviour subspaces (amount;
amount + velocity; amount + velocity + beneficiary; the full vector) × `n_neighbors` 20/35/50 ×
fit rows (20,000 seeded subsample or all). The selected detector uses the **full vector,
50 neighbours, 20,000 fit rows**. Fitting on the subsample both raised validation AP (0.094 versus
0.029 with all rows) and bounds the memory and the per-request neighbour search.

**DBSCAN context adapter (application heuristic).** DBSCAN cannot predict online, so SecFlow freezes
the training core samples (18,145 cores, 17 clusters, 5.0% training noise at `eps = 1.151`,
`min_samples = 10`) and scores each new profile by its distance `d` to the nearest core:

- `d ≤ eps` → 0 (`within_context`);
- `d > eps` → `min(1, (d − eps) / eps)` (`outside_context`; 1 at twice `eps` or further);
- fewer than three earlier requests → 0 (`insufficient_history`).

**Learned weights.** The formula stays `risk = 100 × Σ wᵢ · componentᵢ` with `wᵢ ≥ 0`, `Σ wᵢ = 1`,
so every assessment still splits into per-component contributions. `ml/weights.py` searches all
1,540 weight vectors on a 0.05 lattice (LOF ≥ 0.05) and keeps the one with the highest calibration
average precision. Ties go to the larger rules weight, then IF, then DBSCAN.

| | IF | LOF | DBSCAN | Rules |
|---|---:|---:|---:|---:|
| Learned weight | **0.80** | **0.05** | **0.00** | **0.15** |
| Calibration AP of the component alone | 0.433 | 0.050 | 0.013 | 0.432 |

The combined index reaches calibration AP 0.493 and validation AP 0.668, above every single
component. DBSCAN gets weight 0: its continuous context score adds nothing the other components
do not already rank. It still appears in every assessment and in the cluster graph.

### Bounded search (selected on validation)

| Estimator | Candidates | Selected |
|---|---|---|
| IsolationForest | `n_estimators` 100/200 × `max_samples` 256/512, seed 42 | 200 trees, 256 samples (AP 0.629) |
| LOF | 4 subspaces × `n_neighbors` 20/35/50 × fit rows 20,000/all | full vector, 50, 20,000 (AP 0.094) |
| DBSCAN | `min_samples` 5/10/20 × `eps` at the 90/95/98th percentile of training k-distances | 10, `eps` 1.151 (AP 0.052) |

Every candidate's validation score is stored in `ml/models/training_report.json` and shown on the
Algorithm Results page.

## Rules

Rules are defined in `config/policy.json` (revision 1.1.0) with stable codes, weights, and
families. The rules score is `min(1, Σ weights of triggered rules)`.

| Code | Family | Weight | Corroborates |
|---|---|---:|:-:|
| `AMOUNT_SPIKE` | amount deviation | 0.30 | ✓ |
| `AMOUNT_ELEVATED` | amount deviation | 0.10 | |
| `NEW_BENEFICIARY_HIGH_AMOUNT` | amount deviation | 0.20 | ✓ |
| `VELOCITY_BURST` | velocity | 0.30 | ✓ |
| `RECIPIENT_FANOUT` | velocity | 0.10 | |
| `NEW_DEVICE_AFTER_FAILED_LOGINS` | device & sign-in | 0.35 | ✓ |
| `REPEATED_FAILED_LOGINS` | device & sign-in | 0.15 | ✓ |
| `NEW_DEVICE` | device & sign-in | 0.10 | |
| `FOREIGN_SESSION` | device & sign-in | 0.10 | |
| `SHARED_NEW_BENEFICIARY` | beneficiary network | 0.30 | ✓ |
| `REPEAT_NEW_BENEFICIARY` | beneficiary network | 0.15 | |
| `NEW_BENEFICIARY` | beneficiary network | 0.05 | |
| `PASS_THROUGH_FUNDS` | pass-through funds | 0.30 | ✓ |

`REPEAT_NEW_BENEFICIARY` fires on the second or later payment to the same beneficiary within
24 hours while that relationship is still new.

Corroboration counts **families**, not rules. Several rules derived from the same failed-sign-in
fact count once. IF and LOF agreeing is never counted as corroboration.

History-dependent rules (amount deviation, device change, beneficiary novelty) do not fire for an
account with no history; missing history is never turned into evidence of fraud.

## Combination and action table

`combine_scores` uses the learned weights from the operating point and returns the unrounded total
and per-component contributions, which sum exactly to the total. Decisions use the unrounded value.

| Priority | Condition | Action | `decision_source` |
|---|---|---|---|
| 1 | Active trusted confirmed-fraud evidence at request time | Block (“Confirmed fraud evidence”) | `confirmed_fraud_evidence` |
| 2 | Models unavailable, invalid features, or no history with an amount above €500 or without strong sign-in | Verify | `assessment_safeguard` |
| 3 | Risk ≥ the segment's Block threshold **and** ≥ 2 distinct corroborating families | Block (“Preventive risk block”) | `risk_policy` |
| 4 | Risk ≥ the segment's Verify threshold | Verify | `risk_policy` |
| 5 | Otherwise | Permit | `risk_policy` |

Evidence is active only if it was recorded at or before the request time, is effective then (and
not expired), has `confirmed` as its latest status at that time, comes from a configured trusted
source, and has a blocking evidence type. Expired, retracted, future, pending-review, and
untrusted-source fixtures are in both datasets and never block (tested).

## Operating points per segment

Thresholds are chosen by grid search on validation (Verify 30–95, Block 45–100), separately for
personal and business senders, using model and rules only (no evidence). The search maximises
Verify-or-Block recall subject to: Verify rate ≤ 5%, legitimate false-block rate ≤ 0.1%, and zero
false blocks among predicted blocks. A segment without fraud or without a feasible pair falls back
to the global pair. Cold-start accounts exist only in the test period, so they are not a segment.

| Segment | Validation requests (fraud) | Verify ≥ | Block ≥ | Validation recall (V or B) | Validation verify rate |
|---|---|---:|---:|---:|---:|
| Personal | 7,262 (79) | **80** | **93** | 93.7% | 4.83% |
| Business | 1,527 (6) | **90** | **99** | 66.7% | 1.44% |
| Global (fallback) | 8,789 (85) | 83 | 99 | 88.2% | 3.68% |

The first training run used the earlier grid (Verify ≤ 80, Block ≤ 98). With learned weights no
pair in it met the goals, because corroborated legitimate bursts score above 98. The grid was
widened on validation before the test period was evaluated; the goals did not change.

## Final test results (days 81–100, 20,000 requests, 240 fraud)

Positive = Verify or Block, or Block only. Intervals are 95% cluster-bootstrap intervals (1,000
resamples; a fraud campaign or a single legitimate request is the resampling unit).

| View | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Full policy · Verify or Block | 14.6% (11.0–18.4%) | 76.2% (68.2–83.2%) | 0.245 | 5.40% |
| Full policy · Block only | 97.3% (71/73, Wilson 90.6–99.3%) | 29.6% | 0.454 | 0.01% |
| Model + rules only · Verify or Block | 14.4% | 75.0% | 0.242 | 5.40% |
| Model + rules only · Block only | 96.4% (54/56) | 22.5% | 0.365 | 0.01% |

| Component alone (validation top-5% threshold) | Precision | Recall | Average precision | ROC AUC |
|---|---:|---:|---:|---:|
| IsolationForest | 16.4% | 74.6% | 0.406 | 0.967 |
| LOF (novelty, full vector, 20k rows) | 4.3% | 31.2% | 0.033 | 0.782 |
| DBSCAN context (`outside_context`) | 5.8% | 23.3% | 0.028 | 0.595 |
| Rules | 14.9% | 85.4% | 0.506 | 0.974 |
| Combined risk index | — | — | 0.572 | 0.975 |

| Fraud scenario | Verify or Block | Block | Evidence block | Model + rules only |
|---|---:|---:|---:|---:|
| account_takeover | 90.0% | 37.5% | 20.0% | 90.0% |
| coordinated | 95.0% | 65.0% | 35.0% | 95.0% |
| mule_flow | 100.0% | 42.5% | 2.5% | 100.0% |
| social_engineering | 72.5% | 27.5% | 27.5% | 65.0% |
| velocity | 87.5% | 5.0% | 0.0% | 87.5% |
| invoice_redirect (unseen in training) | 12.5% | 0.0% | 0.0% | 12.5% |

Other measurements:

- **Evidence channel.** 34 confirmed-evidence blocks, all fraud and none legitimate. 206 fraud
  requests had no active evidence; model and rules flagged 72.3% of them.
- **Unseen scenario `invoice_redirect`** (40 requests, 18 campaigns). Single-victim campaigns:
  0 of 5 flagged. Multi-victim campaigns: 5 of 35 flagged (14.3%).
- **Holdout accounts** (40 accounts never fitted on, 2,178 requests, 27 fraud). Fraud recall 70.4%;
  2.7% of their legitimate requests reviewed; none blocked.
- **Drift billers** (471 legitimate payments to billers that first appear on day 81). 14.2% reviewed
  and one blocked (0.21%).
- **Unusual-legitimate requests (1,000).** 30.1% reviewed, 0.10% blocked (one request).
- **Cold-start accounts (40 accounts, 1,415 requests, 9 fraud).** Fraud recall 100%; 28.0% of their
  legitimate requests reviewed; none blocked.
- **Goals.** False-block rate ≤ 0.1%: **met** (0.01%, 2 of 19,760). Verify rate ≤ 5%: **not met**
  (5.89% on test, interval 5.5–6.3%; 3.68% on validation). Block precision ≥ 99%: **not met**
  (97.3%, 71 of 73).
- **Latency (300 single-request assessments, in process).** Mean 5.5 ms, p95 6.2 ms, p99 6.7 ms.

### Honest reading

- **The learned weights lean on IsolationForest and the rules.** On this data they are the two
  strong rankers; LOF keeps the 0.05 minimum and DBSCAN gets 0.
- **Drift costs review volume.** The verify-rate goal held on validation but not on the drifted test
  period. Both preventive false blocks came from legitimate behaviour that looks like fraud (a drift
  biller and an unusual-legitimate request).
- **The beneficiary features helped social engineering** (72.5% flagged) but not the unseen invoice
  scenario. Single-victim invoice redirection looks like an ordinary supplier payment.
- **The demonstration history is sparser than training** (100 requests a day instead of 1,000).
  IsolationForest reads that sparse history as unusual, so the live demo shows a higher review
  share than the test period. The demo is never used to tune anything.
