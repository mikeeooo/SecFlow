# 0001. Import layers: backend, ml, datagen, config

Status: accepted, 2026-11-28.

## Context

SecFlow has four first-party packages. The audit read the import graph (`docs/audit/raw/import-graph.txt`):

- `backend` imports `ml`, `datagen` and `config`.
- `ml` imports `datagen` for its domain types in `datagen.records` and for the dataset loader, and it imports `config`.
- `datagen` imports only `config`.
- `config` imports no other first-party package.

`ml` has 7 module-level imports of `datagen`. If `datagen` sat above `ml`, all 7 would break the contract.

## Decision

The packages form one layer stack, `backend -> ml -> datagen -> config`. A package may import the packages below it and never the ones above it. `[tool.importlinter]` in `pyproject.toml` declares the contract, and CI runs `lint-imports` as a gate.

`ml` uses only the shared domain names from `datagen`: records, the loader and the fraud type names. It does not import `datagen.generator`.

## Consequences

- An import that points up the stack fails CI. A reversed copy of the contract breaks, so the linter does read the graph.
- `scripts/` is not a package and is outside the contract. `scripts/api_contract.py` imports `backend`.
- A new shared name that `ml` needs goes in `datagen/records.py`, not in the generator.
