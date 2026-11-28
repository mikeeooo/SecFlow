# 0004. Check types with mypy --strict

Status: accepted, 2026-11-28.

## Context

The audit ran both checkers in strict mode on the same 32 files (`docs/audit/raw/typecheck_summary.txt`):

| Checker | Errors | Errors from untyped libraries | Wall time |
|---|---:|---:|---|
| mypy 1.19.1 `--strict` | 105 | 13 | 3.8 s |
| pyright 1.1.408 strict | 687 | 552 | 3.6 s |

scikit-learn and joblib ship no type information. pyright's strict mode reports every value that comes from them, so `ml/` would need casts on most model calls. mypy treats those values as `Any` once the imports are marked untyped.

## Decision

Use mypy in strict mode. `[tool.mypy]` in `pyproject.toml` sets `strict = true`, loads the pydantic plugin, and sets `ignore_missing_imports` for `sklearn` and `joblib`. Tests may leave fixtures and helpers unannotated. CI runs `uv run mypy` as a gate.

## Consequences

- Values from scikit-learn are `Any`. Code that returns one converts it at the boundary, for example `np.asarray(..., dtype=np.float64)`. `load_bundle` checks that the loaded object is a `ModelBundle`.
- Speed did not decide this choice. pyright is about 0.2 s faster.
