# Algorithms, policy, and measured results

All three estimators are imported from scikit-learn 1.9.1. Nothing is re-implemented. Every number
below comes from `ml/models/training_report.json` and `artifacts/evaluation.json`.

## Features

`ml/features.py` (`features-2.0.0`) builds two vectors per payment, using only events before the
payment. Amounts are in MDL.

**Payment vector (31 values):** log amount; amount z-score against the sender's 30-day amounts;
log ratio to the 30-day maximum; payments in the last hour and the last 24 hours; log 24-hour sum;
24-hour pace relative to the usual daily rate (normalised by account age when younger than 30
days); unique recipients in 24 hours; no established beneficiary relationship (never paid, or first
paid under 24 hours ago); other senders paying the beneficiary in 24 hours; log beneficiary age;
log device age; failed sign-ins in 24 hours; log hours since the previous payment; log account age;
log incoming funds in 24 hours; foreign session; weak authentication; hour-of-day deviation; log
payments in 30 days; four **beneficiary-side** values; and seven **MIA Plăți Instant** values:

| Feature | Meaning |
|---|---|
| `log_recipient_unique_senders_7d` | Other senders that paid this recipient in the last 7 days |
| `log_recipient_inbound_24h` | Amount the recipient received in the last 24 hours |
| `pair_requests_24h` | This sender's earlier payments to this recipient in 24 h, while the relationship is still new |
| `recipient_new_global` | Nobody paid this recipient before, or the first payment by anyone is under 24 hours old |
| `amount_to_tx_cap_ratio` | Amount ÷ the MIA per-transaction limit of the sender's segment (5,000 MDL personal, 100,000 business) |
| `daily_cumulative_to_cap_ratio` | (Today's earlier payments + this one) ÷ the daily limit (10,000 MDL personal), UTC calendar day |
| `near_cap_count_24h` | Earlier payments in 24 h at or above 90% of the per-transaction limit |
| `alias_rebound_age_hours` | Hours since the sender's phone alias was last rebound to a new SIM or device (capped at 720) |
| `is_request_to_pay` | The payment approves a Request to Pay |
| `qr_merchant_mismatch` | QR scheme whose QR names a merchant other than the recipient |
| `scheme_novelty` | First use of this payment scheme by a sender with earlier payments |

**Account-context vector (8 values, for DBSCAN):** the trailing-24-hour prospective profile. It
covers payment count, sum, unique and new recipients, devices, failed sign-ins, incoming funds, and
the 24-hour sum relative to the usual daily sum.

IDs, names, aliases, labels, fraud and anomaly types, campaign IDs, split assignments, and generator
parameters are never numeric inputs.

## Fitting, score mapping and weight learning

| Step | Data | Labels used? |
|---|---|---|
| Winsorise (0.5/99.5%) and `StandardScaler` | `train.json`, days 1–70, minus holdout accounts (63,431) | No |
| `IsolationForest` | same rows | No |
| `LocalOutlierFactor(novelty=True)` on a behaviour subspace | seeded 20,000-row subsample of those rows | No |
| `DBSCAN` on scaled 24 h profiles | seeded 20,000-profile subsample of train profiles with ≥ 3 prior payments | No |
| IF/LOF empirical CDF mappings | `validation.json` days 71–77, minus holdout (6,311) | No |
| **Combination weights** | same 6,311 rows (116 fraud) | Yes |
| Parameter choice, per-segment Verify/Block thresholds | `validation.json` days 78–85, minus holdout (7,189) | Yes |
| Final report | `test.json`, days 86–100 (15,000) | Yes, once, after freezing |

**Holdout accounts.** Forty established accounts, picked with a fixed seed and without looking at
labels, are removed from every fitting and selection row. Their history still feeds features. The
test period reports their payments as an account-disjoint subset.

**Score direction.** IF uses `-score_samples` and LOF uses `-score_samples` on unseen rows; in both,
a higher value means more anomalous. Each is mapped through the empirical CDF of validation days
71–77 (`searchsorted(side="right") / n`), giving a value in `[0, 1]`.

**LOF.** LOF stays in the index. Training searches 4 behaviour subspaces (amount;
amount + velocity; amount + velocity + beneficiary; the full vector) × `n_neighbors` 20/35/50 ×
fit rows (20,000 seeded subsample or all). The selected detector uses the **amount subspace,
20 neighbours, 20,000 fit rows**. On the same subspace and k, the subsample reached validation AP
0.110, against 0.027 when fitted on every row.

**DBSCAN context adapter (application heuristic).** DBSCAN cannot predict online, so SecFlow freezes
the training core samples (17,942 cores, 131 clusters, 7.3% training noise at `eps = 0.693`,
`min_samples = 5`) and scores each new profile by its distance `d` to the nearest core:

- `d ≤ eps` → 0 (`within_context`);
- `d > eps` → `min(1, (d − eps) / eps)` (`outside_context`; 1 at twice `eps` or further);
- fewer than three earlier payments → 0 (`insufficient_history`).

**Learned weights.** The formula stays `risk = 100 × Σ wᵢ · componentᵢ` with `wᵢ ≥ 0`, `Σ wᵢ = 1`,
so every assessment still splits into per-component contributions. `ml/weights.py` searches all
1,540 weight vectors on a 0.05 lattice (LOF ≥ 0.05) and keeps the one with the highest average
precision on validation days 71–77. Ties go to the larger rules weight, then IF, then DBSCAN.

| | IF | LOF | DBSCAN | Rules |
|---|---:|---:|---:|---:|
| Learned weight | **0.65** | **0.05** | **0.00** | **0.30** |
| AP of the component alone, days 71–77 | 0.642 | 0.062 | 0.025 | 0.710 |

The combined index reaches AP 0.763 on days 71–77 and 0.635 on days 78–85, above every single
component. DBSCAN gets weight 0: its continuous context score adds nothing the other components
do not already rank. It still appears in every assessment and in the cluster graph.

### Bounded search (selected on validation days 78–85)

| Estimator | Candidates | Selected |
|---|---|---|
| IsolationForest | `n_estimators` 100/200 × `max_samples` 256/512, seed 42 | 200 trees, 256 samples (AP 0.532) |
| LOF | 4 subspaces × `n_neighbors` 20/35/50 × fit rows 20,000/all | amount subspace, 20, 20,000 (AP 0.110) |
| DBSCAN | `min_samples` 5/10/20 × `eps` at the 90/95/98th percentile of training k-distances | 5, `eps` 0.693 (AP 0.031) |

Every candidate's validation score is stored in `ml/models/training_report.json` and shown on the
Algorithm Results page.

## Rules

Rules are defined in `config/policy.json` (revision 2.0.0) with stable codes, weights, and
families. Amount parameters are in MDL. The rules score is `min(1, Σ weights of triggered rules)`.

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
| `PASS_THROUGH_FUNDS` (≥ 2,000 MDL received in 24 h) | pass-through funds | 0.30 | ✓ |
| `NEAR_TX_CAP` (≥ 90% of the per-transaction limit) | MIA limit pressure | 0.10 | |
| `DAILY_CAP_PRESSURE` (≥ 80% of the daily limit, new beneficiary) | MIA limit pressure | 0.20 | ✓ |
| `CAP_STRUCTURING` (another near-cap payment within 24 h) | MIA limit pressure | 0.25 | ✓ |
| `RECENT_ALIAS_REBOUND` (alias rebound ≤ 72 h ago) | alias rebound | 0.30 | ✓ |
| `UNKNOWN_RTP_REQUESTER` (Request to Pay, globally new requester) | RTP / QR instrument | 0.25 | ✓ |
| `QR_MERCHANT_MISMATCH` | RTP / QR instrument | 0.35 | ✓ |

`REPEAT_NEW_BENEFICIARY` fires on the second or later payment to the same beneficiary within
24 hours while that relationship is still new. `NEAR_TX_CAP` alone never corroborates: legitimate
planned purchases also sit just under the cap.

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
| 2 | Models unavailable, invalid features, or no history with an amount above 1,000 MDL or without strong sign-in | Verify | `assessment_safeguard` |
| 3 | Risk ≥ the segment's Block threshold **and** ≥ 2 distinct corroborating families | Block (“Preventive risk block”) | `risk_policy` |
| 4 | Risk ≥ the segment's Verify threshold | Verify | `risk_policy` |
| 5 | Otherwise | Permit | `risk_policy` |

Evidence is active only if it was recorded at or before the request time, is effective then (and
not expired), has `confirmed` as its latest status at that time, comes from a configured trusted
source, and has a blocking evidence type. Expired, retracted, future, pending-review, and
untrusted-source fixtures are in the dataset and never block (tested).

## Operating points per segment

Thresholds are chosen by grid search on validation days 78–85 (Verify 30–95, Block 45–100),
separately for personal and business senders, using model and rules only (no evidence). The search
maximises Verify-or-Block recall subject to: Verify rate ≤ 5%, legitimate false-block rate ≤ 0.1%,
and zero false blocks among predicted blocks. A segment without fraud or without a feasible pair
falls back to the global pair. Cold-start accounts exist only in the test period, so they are not a
segment.

| Segment | Selection payments (fraud) | Verify ≥ | Block ≥ | Selection recall (V or B) | Selection verify rate |
|---|---|---:|---:|---:|---:|
| Personal | 3,659 (153) | **71** | **91** | 68.0% | 4.86% |
| Business | 3,530 (0), falls back to global | **69** | **91** | — | — |
| Global | 7,189 (153) | 69 | 91 | 71.9% | 4.55% |

No business fraud occurs in validation (the only business fraud type, `invoice_redirect`, is
test-only), so business senders use the global pair.

## Final test results (`test.json`, days 86–100, 15,000 payments, 360 fraud)

Positive = Verify or Block, or Block only. Intervals are 95% cluster-bootstrap intervals (1,000
resamples; a fraud campaign or a single legitimate payment is the resampling unit).

| View | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Full policy · Verify or Block | 34.4% (28.9–39.5%) | 76.7% (70.8–82.8%) | 0.475 | 3.59% |
| Full policy · Block only | 96.5% (111/115, Wilson 91.4–98.6%) | 30.8% | 0.467 | 0.03% |
| Model + rules only · Verify or Block | 34.2% | 75.8% | 0.471 | 3.59% |
| Model + rules only · Block only | 96.1% (98/102) | 27.2% | 0.424 | 0.03% |

| Component alone (validation top-5% threshold) | Precision | Recall | Average precision | ROC AUC |
|---|---:|---:|---:|---:|
| IsolationForest | 23.4% | 65.6% | 0.507 | 0.942 |
| LOF (novelty, amount subspace, 20k rows) | 7.0% | 14.4% | 0.059 | 0.566 |
| DBSCAN context (`outside_context`) | 8.2% | 16.4% | 0.038 | 0.560 |
| Rules | 32.5% | 75.6% | 0.654 | 0.957 |
| Combined risk index | — | — | 0.657 | 0.967 |

| Fraud type (30 payments each) | Verify or Block | Block | Evidence block | Model + rules only |
|---|---:|---:|---:|---:|
| account_takeover_phishing | 76.7% | 70.0% | 0.0% | 76.7% |
| sim_swap_takeover | 100.0% | 86.7% | 10.0% | 100.0% |
| vishing_impersonation | 70.0% | 13.3% | 3.3% | 70.0% |
| request_to_pay_scam | 100.0% | 50.0% | 26.7% | 100.0% |
| marketplace_scam | 70.0% | 16.7% | 13.3% | 63.3% |
| investment_scam | 36.7% | 0.0% | 0.0% | 36.7% |
| romance_scam | 36.7% | 3.3% | 3.3% | 33.3% |
| money_mule | 96.7% | 33.3% | 10.0% | 96.7% |
| limit_structuring | 100.0% | 33.3% | 0.0% | 100.0% |
| coordinated_beneficiary | 100.0% | 60.0% | 10.0% | 100.0% |
| qr_code_tampering (test only) | 83.3% | 3.3% | 3.3% | 83.3% |
| invoice_redirect (test only) | 50.0% | 0.0% | 0.0% | 50.0% |

Other measurements:

- **Evidence channel.** 24 confirmed-evidence blocks, all fraud and none legitimate. 336 fraud
  payments had no active evidence; model and rules flagged 75.0% of them.
- **Test-only types** (60 payments, 22 campaigns): 66.7% flagged. Single-victim campaigns 50.0%,
  multi-victim campaigns 67.9%.
- **Holdout accounts** (40 accounts never fitted on, 1,365 payments, 30 fraud). Fraud recall 80.0%;
  2.2% of their legitimate payments reviewed; none blocked.
- **Drift billers** (372 legitimate payments to billers first paid in the test period). 14.8%
  reviewed and one blocked (0.27%).
- **Unusual-legitimate payments (750).** 24.5% reviewed, 0.40% blocked (3 payments). Near-limit
  planned purchases (96.7%) and limit increases followed by a large payment (88.9%) are almost
  always reviewed: in the data they look like phishing or vishing at the cap.
- **Routine legitimate payments (13,890).** 2.4% reviewed, one blocked.
- **Cold-start accounts (40 accounts, 678 payments, 30 fraud).** Fraud recall 100%; 14.5% of their
  legitimate payments reviewed; one blocked.
- **Goals.** False-block rate ≤ 0.1%: **met** (0.03%, 4 of 14,640). Verify rate ≤ 5%: **met**
  (4.58%, interval 4.2–5.0%). Block precision ≥ 99%: **not met** (96.5%, 111 of 115).
- **Latency (300 single-request assessments, in process).** Mean 4.5 ms, p95 4.7 ms, p99 4.8 ms.

### Honest reading

- **The learned weights lean on IsolationForest and the rules.** On this data they are the two
  strong rankers; LOF keeps the 0.05 minimum and DBSCAN gets 0.
- **Takeovers and limit abuse are easy here; slow scams are not.** SIM swap, Request-to-Pay scams,
  structuring and coordinated beneficiaries are flagged 100% of the time, because the new MIA
  features describe them directly. Investment and romance scams (36.7%) pay one new person moderate
  or slowly escalating amounts from the victim's own device, which looks like an ordinary
  relationship.
- **Near the cap, legitimate and fraudulent look alike.** Most legitimate near-cap purchases are
  reviewed. The block threshold needs two corroborating families, so they are rarely blocked.
- **The QR mismatch is new information, not a learned pattern.** No QR tampering exists in train
  or validation. The `QR_MERCHANT_MISMATCH` rule still flags 83.3% of the test-only type, but almost
  never blocks it (3.3%), because one family is not enough for a block.
- **Invoice redirection stays hard** (50.0%). A business paying a new supplier an ordinary amount
  from a known device looks like normal business.
- **The app history is thinner than the evaluation history.** The app serves only `test.json`, so
  in the app the first test days have no earlier payments behind them, while this evaluation sees
  days 1–85 as history. Scores and decisions shown in the app for early test payments can therefore
  differ from the ones measured here.
