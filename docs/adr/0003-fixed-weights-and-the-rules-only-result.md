# 0003. Keep the fixed weights and report that rules alone match them

Status: accepted, 2026-11-28. The operator decided: "Report it honestly."

## Context

The risk score combines four components with the fixed weights in `config/policy.json`: IsolationForest 0.65, LOF 0.05, DBSCAN 0.00 and rules 0.30. The original done criterion said the combined score must beat its baselines with non-overlapping 95% intervals.

The audit (`docs/AUDIT.md`, F01) measured the combined score against each component alone on `storage/test.json`:

| Score | Validation AP | Test AP [95% CI] |
|---|---:|---:|
| Combined, fixed weights | 0.719 | 0.668 [0.621, 0.714] |
| Rules only | 0.696 | 0.675 [0.628, 0.721] |
| IsolationForest only | 0.566 | 0.495 [0.445, 0.550] |

Rules alone are the best single component on validation. On test they match the combined score. The paired bootstrap difference of the AP, combined minus rules, is -0.007 [-0.023, 0.009]. With campaign-level resamples it is -0.007 [-0.032, 0.017]. In the leave-one-out ablation, dropping IsolationForest or LOF costs no AP that the bootstrap can tell apart from zero. The anomaly detectors add no measurable average precision on test.

A new search on validation finds better weights: IF 0.40, LOF 0.05, DBSCAN 0.00 and rules 0.55, with validation AP 0.725. The test set has already been seen, so a change chosen now would no longer be measured on unseen data.

## Decision

- Keep the system as it is: DBSCAN, IsolationForest, LOF and the rules, the current weights and thresholds, the committed model bundle and the committed evaluation. Add no model and retrain nothing.
- Replace the done criterion. The combined score must not be significantly worse than the best single component. That holds when the upper end of the paired bootstrap 95% interval of combined minus best is at or above 0. The best component is chosen on validation, so the test labels do not pick it.
- Report the result in `README.md` and `docs/ALGORITHMS.md`: rules alone match the combined score on test, and the anomaly detectors add no measurable AP.

## Consequences

- `backend/tests/test_controls.py` checks the new criterion on every CI run. It also pins the reported numbers: combined 0.668, rules 0.675 and the interval [-0.023, 0.009]. A change to the model or the data fails the test until the reports and docs are updated.
- The same file checks that shuffled test labels score at the fraud prevalence, 0.024.
- CI runs `scripts/experiments.py --check` and puts the full comparison, the rules-only row included, in the job summary.
- The payment-level interval is too narrow for fraud that arrives in campaigns. The test uses it anyway, because the narrower interval is the stricter check for "not worse". The campaign-level interval is reported next to it.
- Any later change to the weights or to the IsolationForest inputs needs a new, unseen test set before it can claim an improvement.
