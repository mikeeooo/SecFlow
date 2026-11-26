# SecFlow

SecFlow analyses instant-payment-style requests and explains what it decides. Each request gets
an anomaly-risk index built from three scikit-learn algorithms (IsolationForest,
LocalOutlierFactor, DBSCAN) and explicit behaviour rules. An evidence-first policy then turns
that index into **Permit**, **Verify**, or **Block**, and every decision shows its arithmetic,
reason codes, and where the action came from.

```text
risk_score = 100 × (0.80 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.15 × rules)
```

The weights are learned on the calibration period (exhaustive search on the simplex, LOF kept at
≥ 0.05), and Verify/Block thresholds are chosen separately for personal and business accounts.

The score is a relative anomaly index, **not a probability of fraud**. A high score alone is never
described as confirmed fraud.

## Quick start

Requirements: Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/) (developed on Python 3.14.3) and
Node.js ≥ 22.22 (developed on Node 24.8). No cloud services or credentials are needed.

```bash
uv sync
```

```bash
npm --prefix frontend install
```

The generated datasets, the trained model bundle, and the evaluation report are committed, so you
can start the app straight away:

```bash
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8001
```

```bash
npm --prefix frontend run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend on port 8001. The simulated
transfer stream starts with the backend, so Live Transfers fills without opening the simulator.

To serve the production build from the backend process instead, run
`npm --prefix frontend run build`, restart uvicorn, and open <http://127.0.0.1:8001>.

## Deploying on Vercel (Services)

`vercel.json` defines one Vercel project with two services:

| Service | Root | What it is | Public path |
|---|---|---|---|
| `app` | `.` | FastAPI backend (`backend.api:app`) | `/api/*` |
| `frontend` | `frontend/` | Vite static build, with an SPA fallback to `index.html` | everything else |

The browser calls the backend on the same domain through relative `/api/...` URLs, so no service
calls another server-side and no bindings are needed. The backend keeps its `/api` prefix because
Vercel passes the original request path through. Offline-only files (the 88 MB training corpus,
cached features, tests, docs, and the frontend source) are excluded from the backend function
bundle.

Run both services together locally with the Vercel CLI:

```bash
npx vercel dev -L
```

On Vercel the backend runs as a function on Fluid compute. The process-memory caveat below matters
more there: instances can be recycled or scaled out, so simulations, reviews, and the live stream
are per instance and short-lived.

## Rebuilding everything from scratch

```bash
uv run python -m backend.cli generate   # both seeded datasets, manifests, then validation
```

```bash
uv run python -m backend.cli train      # fit on days 1-60, map + learn weights on 61-70, select on 71-80
```

```bash
uv run python -m backend.cli evaluate   # one frozen evaluation on days 81-100
```

`generate` is deterministic: the same seeds reproduce byte-identical files (checked by SHA-256 in
`manifest.json`). `uv run python -m backend.cli validate` re-checks counts, partitions, campaign
integrity, hashes, and cross-dataset fingerprints at any time.

## Checks

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
npm --prefix frontend test
```

```bash
npm --prefix frontend run typecheck
```

```bash
npm --prefix frontend run build
```

## What is in the box

| Path | Contents |
|---|---|
| `storage/generator.py`, `storage/identities.py` | Seeded generator for both datasets (400 named accounts) |
| `storage/data/training/` | 100,000-request corpus (seed 42), labels, splits, evidence, manifest |
| `storage/data/demo/` | Independent 10,000-request demonstration history (seed 43) |
| `storage/history.py`, `storage/evidence.py`, `storage/memory.py` | Time-ordered history index, trusted evidence lookup, in-process state |
| `ml/features.py` | The one leakage-safe feature function used offline and live |
| `ml/algorithms/` | scikit-learn adapters: preprocessing, IF, LOF (novelty), DBSCAN context |
| `ml/policy.py`, `config/policy.json` | Rules, pure score combination, evidence-first action table |
| `ml/training.py`, `ml/evaluation.py` | Bounded search, threshold selection, frozen evaluation |
| `ml/models/` | Model bundle, metadata, operating point, training report |
| `artifacts/evaluation.{json,md}` | Measured evaluation output |
| `backend/` | FastAPI app, service, stream producer, CLI, tests |
| `frontend/` | React + TypeScript + Vite + Tailwind interface and tests |
| `docs/` | Architecture, algorithms, dataset provenance, interface design |

## Measured results (final test period, synthetic data)

These come from `artifacts/evaluation.md`. They describe this generated dataset only and are not
evidence of real-world detection or fraud reduction.

| Metric | Value |
|---|---|
| Fraud sent to Verify or Block | 76.2% (183 of 240; 95% CI 68.2–83.2%) |
| Fraud blocked | 29.6% (71 of 240; 34 by confirmed evidence, 37 preventive) |
| Block precision | 97.3% over 73 blocks (95% CI 90.6–99.3%; goal 99%: **not met**) |
| Legitimate false-block rate | 0.01% (2 of 19,760) |
| Verify rate | 5.89% (validation goal ≤ 5%: **not met** on test; 3.68% on validation) |
| Model + rules only, Verify or Block | 75.0% |
| Unseen scenario `invoice_redirect` | 12.5% flagged (5 of 40) |
| Holdout accounts (never fitted on) | 70.4% fraud recall, 2.7% of legitimate requests reviewed |
| Single-request latency | p50 5.5 ms, p95 6.2 ms (in process) |

### What the risk percentage is worth

The score ranks how unusual a request is; its yield is measured per score band on the labelled test
period (`score_bands` in `artifacts/evaluation.json`, shown on every investigation and on Algorithm
Results):

| Score band | Requests | Fraud | Observed fraud rate | Zone |
|---|---:|---:|---:|---|
| 0–50 | 10,955 | 0 | 0% | Permit |
| 50–70 | 4,645 | 2 | 0.04% | Permit |
| 70–80 | 2,448 | 22 | 0.9% | Permit |
| 80–83 | 809 | 15 | 1.9% | Verify (personal) |
| 83–90 | 946 | 74 | 7.8% | Verify |
| 90–99 | 173 | 104 | 60.1% | Verify / Block |
| 99–100 | 24 | 23 | 95.8% | Block zone |

Base fraud rate is 1.2%, so a request scoring 90–99 is about 50× more likely to be fraud than an
average one. Flagging everything from 83 up catches 83.8% of fraud while touching 5.71% of traffic.

See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for the per-component and per-scenario breakdown and
the tradeoffs behind these numbers.

## Limitations to know before you rely on it

- **Process memory.** Simulations and the live stream live in one backend process. A restart
  discards them; the baseline files and models remain.
- **Review history is durable.** Every approve or reject is written to
  `storage/state/review_history.sqlite3` (override with `SECFLOW_STATE_DIR`; on Vercel it goes to
  `/tmp`, which is per instance). After a restart, reviews of baseline requests are re-applied and
  the Review History page still lists every decision, including reviews of simulated requests. Multiple workers or instances
  would each hold separate, unshared state. Run a single process (`uvicorn` without `--workers`).
- **Synthetic everything.** Accounts, counterparties, payments, and security evidence are
  generated. Names are plausible but belong to no verified person or business.
- **The demo history is sparser** (100 requests per day versus 1,000 in the training corpus), so
  its Verify share (≈19%) is higher than the test-period rate: IsolationForest, which carries most
  of the learned weight, reads the sparse history as unusual. It is product history, not a
  benchmark.
- **Preventive blocks are rare by design.** The validated operating point keeps ambiguous
  high-risk cases in Verify instead of blocking them; see the tradeoff in the docs.
- **No authentication.** The reviewer name is a display value for this local workflow.
