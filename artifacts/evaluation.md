# SecFlow evaluation report

Generated 2026-11-28T05:56:06Z · model `secflow-sk-3.0-4b97752f19` · policy `secflow-evidence-first-3.0.0+c4573fe9`

> Synthetic final-test file storage/test.json (days 86-100, 15,000 assessed MIA instant payments in MDL). Results describe this generated data only; they are not evidence of real-world detection.

Thresholds chosen on validation and frozen before test: **Verify ≥ 68**, **Block ≥ 86** with at least two corroborating rule families.

## Decisions on the final test period

| View | Precision | Recall | F1 | FPR | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With evidence · Verify or Block | 35.3% | 78.6% | 0.487 | 3.55% | 283 | 519 | 77 | 14121 |
| With evidence · Block only | 96.9% | 34.7% | 0.511 | 0.03% | 125 | 4 | 235 | 14636 |
| Model + rules only · Verify or Block | 35.1% | 78.1% | 0.484 | 3.55% | 281 | 519 | 79 | 14121 |
| Model + rules only · Block only | 96.6% | 31.7% | 0.477 | 0.03% | 114 | 4 | 246 | 14636 |

Verify rate 4.49% · block rate 0.86% · legitimate false-block rate 0.027% · 129 blocks.
95% intervals (Wilson score interval; it treats payments as independent, so it is too narrow for fraud that arrives in campaigns): recall [0.7408, 0.8253], precision [0.3206, 0.3866], verify rate [0.0417, 0.0483].

## Components alone

| Component | Threshold | Precision | Recall | FPR | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| isolation_forest | 0.950 | 22.9% | 63.1% | 5.23% | 0.495 | 0.937 |
| lof | 0.950 | 4.4% | 10.3% | 5.49% | 0.051 | 0.574 |
| dbscan | 0.000 | 7.7% | 14.2% | 4.19% | 0.035 | 0.551 |
| rules | 0.150 | 34.0% | 86.1% | 4.11% | 0.675 | 0.960 |
| combined risk score | — | — | — | — | 0.668 | 0.963 |

## Recall per fraud type

| Fraud type | Test only | Payments | Verify or Block | Block | Evidence block | Model + rules only |
|---|---|---:|---:|---:|---:|---:|
| account_takeover_phishing |  | 30 | 93.3% | 70.0% | 0.0% | 93.3% |
| sim_swap_takeover |  | 30 | 100.0% | 86.7% | 10.0% | 100.0% |
| vishing_impersonation |  | 30 | 76.7% | 20.0% | 3.3% | 76.7% |
| request_to_pay_scam |  | 30 | 100.0% | 56.7% | 26.7% | 100.0% |
| marketplace_scam |  | 30 | 73.3% | 20.0% | 13.3% | 66.7% |
| investment_scam |  | 30 | 50.0% | 0.0% | 0.0% | 50.0% |
| romance_scam |  | 30 | 40.0% | 3.3% | 3.3% | 40.0% |
| money_mule |  | 30 | 96.7% | 26.7% | 10.0% | 96.7% |
| limit_structuring |  | 30 | 100.0% | 46.7% | 0.0% | 100.0% |
| coordinated_beneficiary |  | 30 | 100.0% | 73.3% | 10.0% | 100.0% |
| qr_code_tampering | yes | 30 | 80.0% | 13.3% | 3.3% | 80.0% |
| invoice_redirect | yes | 30 | 33.3% | 0.0% | 0.0% | 33.3% |

## Unusual but legitimate payments

| Anomaly type | Payments | Verify or Block | Block |
|---|---:|---:|---:|
| business_settlement | 29 | 34.5% | 0.0% |
| diaspora_remittance_spend | 44 | 52.3% | 0.0% |
| group_payment_new_payee | 113 | 46.9% | 0.0% |
| holiday_spike | 53 | 24.5% | 0.0% |
| household_repeat_payments | 106 | 4.7% | 0.0% |
| legit_alias_change | 20 | 10.0% | 0.0% |
| limit_increase_then_large_payment | 18 | 94.4% | 5.6% |
| near_limit_planned_purchase | 30 | 100.0% | 3.3% |
| new_device_legit | 37 | 35.1% | 0.0% |
| new_merchant_qr | 18 | 0.0% | 0.0% |
| night_payment | 23 | 4.3% | 0.0% |
| payroll_burst | 106 | 9.4% | 0.0% |
| salary_day_spike | 99 | 1.0% | 0.0% |
| travel_abroad | 54 | 46.3% | 0.0% |

## Subsets

| Subset | Requests | Fraud | Recall (V or B) | Legit review rate | Legit false-block rate |
|---|---:|---:|---:|---:|---:|
| Unusual legitimate | 750 | 0 | n/a | 26.80% | 0.267% |
| Routine legitimate | 13,890 | 0 | n/a | 2.26% | 0.014% |
| Cold-start accounts | 678 | 30 | 100.0% | 23.15% | 0.309% |
| Existing accounts | 14,322 | 330 | 76.7% | 2.61% | 0.014% |
| Test-only fraud types | 60 | 60 | 56.7% | n/a | n/a |

## Evidence channel

24 confirmed-evidence blocks (24 fraud, 0 legitimate). Model and rules flagged 77.1% of the 336 fraud payments without evidence.

## Goals

- Legitimate false-block rate <= 0.1%: measured 0.0003, **met**
- Verify rate <= 5%: measured 0.0449, **met**
- Block precision >= 99%: measured 0.9690, **not met**

## Latency

300 single payments: mean 4.548 ms, p95 4.685 ms, p99 4.786 ms.

Scores are anomaly-risk indices, not probabilities of fraud.
