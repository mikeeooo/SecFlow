# SecFlow evaluation report

Generated 2026-11-27T16:27:32Z · model `secflow-sk-1.1-a7a420d885` · policy `secflow-evidence-first-1.1.0+bf92c73f`

> Synthetic final-test period (days 81-100) of the 100,000-request training corpus. Results describe this generated data only; they are not evidence of real-world detection or fraud reduction.

Operating point (selected on validation, frozen before test): **Verify ≥ 83**, **Block ≥ 99** with at least two corroborating behaviour families (`validated`).

## Splits

| Partition | Requests | Fraud | Fraud share |
|---|---:|---:|---:|
| train | 60,000 | 600 | 1.00% |
| calibration | 10,000 | 100 | 1.00% |
| validation | 10,000 | 100 | 1.00% |
| test | 20,000 | 240 | 1.20% |

## Combined policy on the final test period

Positive definitions: *Verify or Block* — A request counts as flagged when the final action is Verify or Block. *Block only* — A request counts as flagged only when the final action is Block.

| View | Precision | Recall | F1 | FPR | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With evidence · Verify or Block | 14.6% | 76.2% | 0.245 | 5.40% | 183 | 1068 | 57 | 18692 |
| With evidence · Block only | 97.3% | 29.6% | 0.454 | 0.01% | 71 | 2 | 169 | 19758 |
| Model + rules only · Verify or Block | 14.4% | 75.0% | 0.242 | 5.40% | 180 | 1068 | 60 | 18692 |
| Model + rules only · Block only | 96.4% | 22.5% | 0.365 | 0.01% | 54 | 2 | 186 | 19758 |

Verify rate 5.89% · Block rate 0.36% · legitimate false-block rate 0.010% · block precision 95% CI [0.9055, 0.9925] over 73 blocks.

## Components alone

Flag thresholds come from the validation period (top 5% review budget; DBSCAN flags `outside_context`).

| Component | Threshold | Precision | Recall | FPR | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| isolation_forest | 0.967 | 16.4% | 74.6% | 4.63% | 0.406 | 0.967 |
| lof | 0.962 | 4.3% | 31.2% | 8.53% | 0.033 | 0.782 |
| dbscan | 0.000 | 5.8% | 23.3% | 4.61% | 0.028 | 0.595 |
| rules | 0.150 | 14.9% | 85.4% | 5.91% | 0.506 | 0.974 |
| combined risk score | — | — | — | — | 0.572 | 0.975 |

## Per scenario (final test)

| Scenario | Kind | Requests | Verify or Block | Block | Evidence block | Model-only Verify or Block |
|---|---|---:|---:|---:|---:|---:|
| account_recovery | legitimate | 31 | 22.6% | 0.0% | 0.0% | 22.6% |
| account_takeover | fraud | 40 | 90.0% | 37.5% | 20.0% | 90.0% |
| business_settlement | legitimate | 47 | 0.0% | 0.0% | 0.0% | 0.0% |
| coordinated | fraud | 40 | 95.0% | 65.0% | 35.0% | 95.0% |
| household_repeat | legitimate | 192 | 6.8% | 0.0% | 0.0% | 6.8% |
| invoice_redirect | fraud | 40 | 12.5% | 0.0% | 0.0% | 12.5% |
| large_planned_purchase | legitimate | 52 | 48.1% | 0.0% | 0.0% | 48.1% |
| mule_flow | fraud | 40 | 100.0% | 42.5% | 2.5% | 100.0% |
| new_device_legitimate | legitimate | 74 | 39.2% | 0.0% | 0.0% | 39.2% |
| new_recipient_legitimate | legitimate | 56 | 44.6% | 0.0% | 0.0% | 44.6% |
| payroll_burst | legitimate | 277 | 3.6% | 0.0% | 0.0% | 3.6% |
| routine_payment | legitimate | 18,760 | 4.1% | 0.0% | 0.0% | 4.1% |
| shared_group_payment | legitimate | 140 | 72.9% | 0.7% | 0.0% | 72.9% |
| social_engineering | fraud | 40 | 72.5% | 27.5% | 27.5% | 65.0% |
| travel | legitimate | 131 | 69.5% | 0.0% | 0.0% | 69.5% |
| velocity | fraud | 40 | 87.5% | 5.0% | 0.0% | 87.5% |

## What the risk score is worth: observed fraud rate per score band (final test)

Base fraud rate 1.20% over 20,000 scored requests. *Recall ≥ band* and *precision ≥ band* describe flagging every request at or above the band's lower edge.

| Score band | Requests | Fraud | Fraud rate (95% CI) | Lift | Recall ≥ band | Precision ≥ band | Flag rate ≥ band |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–10 | 1,918 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 1.2% | 100.00% |
| 10–20 | 2,196 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 1.3% | 90.41% |
| 20–30 | 2,306 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 1.5% | 79.43% |
| 30–40 | 2,207 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 1.8% | 67.90% |
| 40–50 | 2,328 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 2.1% | 56.87% |
| 50–70 | 4,645 | 2 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 2.7% | 45.23% |
| 70–80 | 2,448 | 22 | 0.9% (0.6–1.4%) | 0.7× | 99.2% | 5.4% | 22.00% |
| 80–83 | 809 | 15 | 1.9% (1.1–3.0%) | 1.5× | 90.0% | 11.1% | 9.76% |
| 83–90 | 946 | 74 | 7.8% (6.3–9.7%) | 6.5× | 83.8% | 17.6% | 5.71% |
| 90–99 | 173 | 104 | 60.1% (52.7–67.1%) | 50.1× | 52.9% | 64.5% | 0.98% |
| 99–100 | 24 | 23 | 95.8% (79.8–99.3%) | 79.9× | 9.6% | 95.8% | 0.12% |

Observed on the labelled synthetic final-test period. The fraud rate of a band is how often a request with that score really was fraud; it is an empirical rate, not a calibrated probability for any single request.

## Evidence channel versus model-only detection

- Confirmed-evidence blocks: 34 (34 fraud, 0 legitimate).
- Fraud requests without any active evidence: 206; model + rules flagged 72.3% of them (Verify or Block).
- All security evidence is simulated. The evidence channel is reported separately so it cannot hide weak model-only detection.

## Subsets

| Subset | Requests | Fraud | Verify or Block recall | Block recall | Legit review rate | Legit false-block rate |
|---|---:|---:|---:|---:|---:|---:|
| Unusual legitimate | 1,000 | 0 | n/a | n/a | 30.10% | 0.100% |
| Routine legitimate | 18,760 | 0 | n/a | n/a | 4.08% | 0.005% |
| Cold-start accounts (40) | 1,415 | 9 | 100.0% | 44.4% | 27.95% | 0.000% |
| Existing accounts | 18,585 | 231 | 75.3% | 29.0% | 3.67% | 0.011% |

## Validation goals (targets to assess, not promises)

- Legitimate false-block rate <= 0.1%: measured 0.0001 — **met**
- Verify rate <= 5%: measured 0.0589 — **not met**
- Block precision >= 99%: measured 0.9726 — **not established** (Fewer than 100 predicted blocks: the sample cannot establish 99% precision either way.)

## Latency

300 single-request assessments (single-request path: features + IF + LOF + DBSCAN context + rules + evidence + policy, in-process): mean 5.57 ms, p50 5.432 ms, p95 6.239 ms, p99 10.398 ms.

## Decision sources

- Block/confirmed_fraud_evidence: 34
- Block/risk_policy: 39
- Permit/risk_policy: 18,749
- Verify/assessment_safeguard: 14
- Verify/risk_policy: 1,164

## Learned weights and segment thresholds

- Weights (calibration AP search): isolation_forest 0.80, lof 0.05, dbscan 0.00, rules 0.15
- Segment thresholds (validation): personal: Verify ≥ 80, Block ≥ 93; business: Verify ≥ 90, Block ≥ 99

## Confidence intervals (95%)

- Verify-or-Block precision: 11.0% – 18.4%
- Verify-or-Block recall: 68.2% – 83.2%
- Verify rate: 5.5% – 6.3%
- Method: cluster bootstrap, 1000 resamples, seed 42; a fraud campaign or one legitimate request is the resampling unit; 2.5th-97.5th percentiles

## Unseen scenario, holdout accounts and drift

- invoice_redirect (40 requests, 18 campaigns): flagged 12.5%; single-victim 0.0%, multi-victim 14.3%
- Holdout accounts (40 accounts, 2178 requests, 27 fraud): recall 70.4%, verify rate 3.1%
- Drift billers (471 requests): verify rate 14.2%, false-block rate 0.21%

Scores are anomaly-risk indices, not probabilities of fraud. A high score alone is never treated as confirmed fraud.
