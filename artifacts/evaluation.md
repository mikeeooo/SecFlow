# SecFlow evaluation report

Generated 2026-11-27T20:44:28Z · model `secflow-sk-2.0-910679dc58` · policy `secflow-evidence-first-2.0.0+91b14a52`

> Synthetic final-test file storage/test.json (days 86-100, 15,000 assessed MIA instant payments in MDL). Results describe this generated data only; they are not evidence of real-world detection or fraud reduction.

Operating point (selected on validation, frozen before test): **Verify ≥ 69**, **Block ≥ 91** with at least two corroborating behaviour families (`validated`).

## Splits

| Partition | Requests | Fraud | Fraud share |
|---|---:|---:|---:|
| train | 70,000 | 1500 | 2.14% |
| validation_mapping | 7,000 | 127 | 1.81% |
| validation_selection | 8,000 | 173 | 2.16% |
| test | 15,000 | 360 | 2.40% |

## Combined policy on the final test period

Positive definitions: *Verify or Block* — A payment counts as flagged when the final action is Verify or Block. *Block only* — A payment counts as flagged only when the final action is Block.

| View | Precision | Recall | F1 | FPR | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With evidence · Verify or Block | 34.4% | 76.7% | 0.475 | 3.59% | 276 | 526 | 84 | 14114 |
| With evidence · Block only | 96.5% | 30.8% | 0.467 | 0.03% | 111 | 4 | 249 | 14636 |
| Model + rules only · Verify or Block | 34.2% | 75.8% | 0.471 | 3.59% | 273 | 526 | 87 | 14114 |
| Model + rules only · Block only | 96.1% | 27.2% | 0.424 | 0.03% | 98 | 4 | 262 | 14636 |

Verify rate 4.58% · Block rate 0.77% · legitimate false-block rate 0.027% · block precision 95% CI [0.914, 0.9864] over 115 blocks.

## Components alone

Flag thresholds come from the validation period (top 5% review budget; DBSCAN flags `outside_context`).

| Component | Threshold | Precision | Recall | FPR | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| isolation_forest | 0.956 | 23.4% | 65.6% | 5.28% | 0.507 | 0.942 |
| lof | 0.961 | 7.0% | 14.4% | 4.73% | 0.059 | 0.566 |
| dbscan | 0.000 | 8.2% | 16.4% | 4.52% | 0.038 | 0.560 |
| rules | 0.300 | 32.5% | 75.6% | 3.86% | 0.654 | 0.957 |
| combined risk score | — | — | — | — | 0.657 | 0.967 |

## Recall per fraud type (final test)

| Fraud type | Test only | Payments | Campaigns | Verify or Block | Block | Evidence block | Model-only Verify or Block |
|---|---|---:|---:|---:|---:|---:|---:|
| account_takeover_phishing |  | 30 | 20 | 76.7% | 70.0% | 0.0% | 76.7% |
| sim_swap_takeover |  | 30 | 12 | 100.0% | 86.7% | 10.0% | 100.0% |
| vishing_impersonation |  | 30 | 15 | 70.0% | 13.3% | 3.3% | 70.0% |
| request_to_pay_scam |  | 30 | 7 | 100.0% | 50.0% | 26.7% | 100.0% |
| marketplace_scam |  | 30 | 8 | 70.0% | 16.7% | 13.3% | 63.3% |
| investment_scam |  | 30 | 7 | 36.7% | 0.0% | 0.0% | 36.7% |
| romance_scam |  | 30 | 8 | 36.7% | 3.3% | 3.3% | 33.3% |
| money_mule |  | 30 | 9 | 96.7% | 33.3% | 10.0% | 96.7% |
| limit_structuring |  | 30 | 9 | 100.0% | 33.3% | 0.0% | 100.0% |
| coordinated_beneficiary |  | 30 | 7 | 100.0% | 60.0% | 10.0% | 100.0% |
| qr_code_tampering | yes | 30 | 8 | 83.3% | 3.3% | 3.3% | 83.3% |
| invoice_redirect | yes | 30 | 14 | 50.0% | 0.0% | 0.0% | 50.0% |

## Unusual but legitimate payments per anomaly type (final test)

| Anomaly type | Payments | Verify or Block | Block |
|---|---:|---:|---:|
| business_settlement | 29 | 27.6% | 0.0% |
| diaspora_remittance_spend | 44 | 61.4% | 2.3% |
| group_payment_new_payee | 113 | 31.9% | 0.0% |
| holiday_spike | 53 | 18.9% | 0.0% |
| household_repeat_payments | 106 | 3.8% | 0.0% |
| legit_alias_change | 20 | 45.0% | 0.0% |
| limit_increase_then_large_payment | 18 | 88.9% | 5.6% |
| near_limit_planned_purchase | 30 | 96.7% | 3.3% |
| new_device_legit | 37 | 35.1% | 0.0% |
| new_merchant_qr | 18 | 0.0% | 0.0% |
| night_payment | 23 | 0.0% | 0.0% |
| payroll_burst | 106 | 12.3% | 0.0% |
| salary_day_spike | 99 | 1.0% | 0.0% |
| travel_abroad | 54 | 38.9% | 0.0% |
| routine (no anomaly) | 13,890 | 2.4% | 0.0% |

## What the risk score is worth: observed fraud rate per score band (final test)

Base fraud rate 2.40% over 15,000 scored requests. *Recall ≥ band* and *precision ≥ band* describe flagging every request at or above the band's lower edge.

| Score band | Requests | Fraud | Fraud rate (95% CI) | Lift | Recall ≥ band | Precision ≥ band | Flag rate ≥ band |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–10 | 1,606 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 2.4% | 100.00% |
| 10–20 | 2,250 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 2.7% | 89.29% |
| 20–30 | 2,133 | 0 | 0.0% (0.0–0.2%) | 0.0× | 100.0% | 3.2% | 74.29% |
| 30–40 | 2,190 | 1 | 0.0% (0.0–0.3%) | 0.0× | 100.0% | 4.0% | 60.07% |
| 40–50 | 2,076 | 5 | 0.2% (0.1–0.6%) | 0.1× | 99.7% | 5.3% | 45.47% |
| 50–69 | 3,837 | 70 | 1.8% (1.5–2.3%) | 0.8× | 98.3% | 7.5% | 31.63% |
| 69–70 | 93 | 9 | 9.7% (5.2–17.4%) | 4.0× | 78.9% | 31.3% | 6.05% |
| 70–80 | 543 | 84 | 15.5% (12.7–18.8%) | 6.4× | 76.4% | 33.7% | 5.43% |
| 80–90 | 161 | 86 | 53.4% (45.7–61.0%) | 22.3× | 53.1% | 70.2% | 1.81% |
| 90–91 | 9 | 7 | 77.8% (45.3–93.7%) | 32.4× | 29.2% | 94.6% | 0.74% |
| 91–100 | 102 | 98 | 96.1% (90.3–98.5%) | 40.0× | 27.2% | 96.1% | 0.68% |

Observed on the labelled synthetic final-test period. The fraud rate of a band is how often a request with that score really was fraud; it is an empirical rate, not a calibrated probability for any single request.

## Evidence channel versus model-only detection

- Confirmed-evidence blocks: 24 (24 fraud, 0 legitimate).
- Fraud requests without any active evidence: 336; model + rules flagged 75.0% of them (Verify or Block).
- All security evidence is simulated. The evidence channel is reported separately so it cannot hide weak model-only detection.

## Subsets

| Subset | Requests | Fraud | Verify or Block recall | Block recall | Legit review rate | Legit false-block rate |
|---|---:|---:|---:|---:|---:|---:|
| Unusual legitimate | 750 | 0 | n/a | n/a | 24.53% | 0.400% |
| Routine legitimate | 13,890 | 0 | n/a | n/a | 2.43% | 0.007% |
| Cold-start accounts (40) | 678 | 30 | 100.0% | 43.3% | 14.51% | 0.154% |
| Existing accounts | 14,322 | 330 | 74.5% | 29.7% | 3.06% | 0.021% |

## Validation goals (targets to assess, not promises)

- Legitimate false-block rate <= 0.1%: measured 0.0003 — **met**
- Verify rate <= 5%: measured 0.0458 — **met**
- Block precision >= 99%: measured 0.9652 — **not met**

## Latency

300 single-request assessments (single-request path: features + IF + LOF + DBSCAN context + rules + evidence + policy, in-process): mean 4.544 ms, p50 4.537 ms, p95 4.653 ms, p99 4.795 ms.

## Decision sources

- Block/confirmed_fraud_evidence: 24
- Block/risk_policy: 91
- Permit/risk_policy: 14,198
- Verify/assessment_safeguard: 18
- Verify/risk_policy: 669

## Learned weights and segment thresholds

- Weights (AP search on validation days 71-77): isolation_forest 0.65, lof 0.05, dbscan 0.00, rules 0.30
- Segment thresholds (validation days 78-85): personal: Verify ≥ 71, Block ≥ 91; business: Verify ≥ 69, Block ≥ 91

## Confidence intervals (95%)

- Verify-or-Block precision: 28.9% – 39.5%
- Verify-or-Block recall: 70.8% – 82.8%
- Verify rate: 4.2% – 5.0%
- Method: cluster bootstrap, 1000 resamples, seed 42; a fraud campaign or one legitimate payment is the resampling unit; 2.5th-97.5th percentiles

## Test-only fraud types, holdout accounts and drift

- Test-only fraud types qr_code_tampering, invoice_redirect (60 payments, 22 campaigns): flagged 66.7%; single-victim 50.0%, multi-victim 67.9%
- Holdout accounts (40 accounts, 1365 requests, 30 fraud): recall 80.0%, verify rate 3.4%
- Drift billers (372 requests): verify rate 14.8%, false-block rate 0.27%

Scores are anomaly-risk indices, not probabilities of fraud. A high score alone is never treated as confirmed fraud.
