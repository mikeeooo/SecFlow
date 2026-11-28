# Architecture and data flow

```text
           offline (backend/cli.py)                            online (one uvicorn process)
┌──────────────────────────────────────────┐      ┌───────────────────────────────────────────────────┐
│ generate ─► storage/{train,validation,   │      │ startup: load test.json without labels, the model │
│              test}.json + manifest       │      │          bundle, the policy and the thresholds;   │
│ validate ─► dataset checks               │      │          assess all 15,000 test payments          │
│ train    ─► fit IF, LOF, DBSCAN on train;│ ml/  │ request ─► MIA limit check ─► sign-in events      │
│              CDF references + threshold  │models│          ─► Assessor (features, detectors, rules, │
│              grid on validation          │─────►│             risk, evidence, decision table)       │
│ evaluate ─► frozen run on test.json      │      │          ─► record the attempt (+ settle if Permit)│
│              ─► artifacts/evaluation.*   │      │ stream task ─► same path                          │
└──────────────────────────────────────────┘      └─────────────────────▲─────────────────────────────┘
                                                                        │ /api (JSON)
                                                  ┌─────────────────────┴─────────────────────────────┐
                                                  │ React app: Overview, Live Transfers, Transactions, │
                                                  │ Accounts, Investigation, Simulator, Algorithms     │
                                                  └───────────────────────────────────────────────────┘
```

## Code layout

| Path | Role |
|---|---|
| `storage/` | Data only: `train.json`, `validation.json`, `test.json` (see [DATASET.md](DATASET.md)) |
| `datagen/` | Generator, identities, runtime records, loader and validator, `manifest.json` |
| `config/` | `settings.py` (paths and stream timing), `policy.json` (weights, rules, goals, MIA limits), `limits.py` |
| `ml/history.py` | Time-ordered history index: every query looks only at events before a key |
| `ml/features.py` | The feature function used by training, evaluation and the app |
| `ml/model.py` | Preprocessor, CDF mapping, DBSCAN context and the saved `ModelBundle` |
| `ml/policy.py` | Policy loading, rules, the risk score and the decision table |
| `ml/evidence.py` | Trusted security evidence and when it is active |
| `ml/assess.py` | `Assessor`: the one path from a payment to a decision |
| `ml/training.py`, `ml/evaluation.py` | Training with threshold search, and the frozen test evaluation |
| `ml/models/` | Model bundle, metadata, operating point, training report |
| `backend/` | FastAPI app, service, views, analytics, stream producer, CLI, tests |
| `scripts/api_contract.py` | The API shape pin the frontend depends on |

## One assessment path

`Assessor.assess_batch` in `ml/assess.py` is the only way from a payment to a decision. The app's
startup, each simulation, the stream and the evaluation all call it.

1. `build_features` reads only events whose ordering key is lower than the payment's key.
2. `ModelBundle.score_batch` runs the preprocessor, IsolationForest, LOF and the DBSCAN context once
   per payment.
3. `evaluate_rules` turns the facts into reason codes and corroborating families.
4. `combine_scores` adds the four weighted parts.
5. `EvidenceStore.active_for` returns the trusted evidence active at the payment time.
6. `decide_action` applies the decision table.

[ALGORITHMS.md](ALGORITHMS.md) explains each step and the measured results.

## Ordering and leakage

Every event gets the key `epoch_seconds × 10^6 + sequence`. Events in the same second sort sign-in
first, then payment, then request, then by identifier. Queries use `bisect_left` on the key, so
adding later events never changes the features of an earlier payment. History counts exclude the
payment being assessed.

Every request counts as an attempt, whatever its decision, so it feeds the velocity and
beneficiary features. Only completed flows (a settled payment or an incoming credit) feed the
incoming-funds features. A blocked or pending simulation is therefore an attempt, but no one
receives its money.

## State in the app

`SecFlowService` in `backend/service.py` keeps everything in memory: the history index, every
assessment record, simulated counterparties and the review list. Nothing is written to disk, so a
restart starts again from `test.json`.

- One lock serialises simulations and reviews. FastAPI runs requests in a thread pool, and the
  stream runs `simulate` in a thread, so the history index needs it.
- `POST /api/simulations` hashes the payload. The same ID with the same payload returns the stored
  assessment (`200`, `idempotent_replay: true`). The same ID with another payload returns `409`.
- A review records its own entry and changes only the payment outcome
  (`approved_after_review` or `rejected_after_review`). The automated decision and score never
  change. A second review of the same payment returns `409`.

The state is per process. Run one backend process; extra workers or Vercel instances each have
their own copy.

## Simulation clock

The baseline is the 15 days of `test.json`. The simulation clock starts one hour after the last
test event and then follows real time. Sign-in events and the request of a simulation are stamped
after the latest event, so the history stays in order.

## Live stream

`backend/stream.py` runs one producer task per process. The task starts with the server unless
`SECFLOW_AUTOSTART_STREAM=false`. It submits one payment every 2 seconds, or 5, 10 or 20 payments
every 4 seconds. Browsers poll `GET /api/live-transfers?after=<cursor>` and call only start, stop
or mode, so extra tabs never add submissions. The producer checks `running` before each submission,
so Stop takes effect within one payment.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Readiness, data file, model and policy versions, memory state |
| GET | `/api/overview`, `/api/dashboard` | Totals, volumes, trend, recent activity, review queue, monitoring buckets |
| GET | `/api/transactions` | Filter (`q`, `decision`, `decision_source`, `origin`, `review_status`, dates, score) and paginate |
| GET | `/api/accounts`, `/api/accounts/{id}` | Account search, behaviour profile, DBSCAN context, linked payments |
| GET | `/api/parties` | Account and counterparty search for the simulator |
| GET | `/api/simulator/options` | Known devices, recent recipients, beneficiaries with active evidence, schemes and limits |
| POST | `/api/simulations` | Assess and record one simulated payment |
| GET | `/api/live-transfers` | Simulated assessments after a cursor, plus the stream status |
| GET, POST | `/api/simulation-stream`, `/start`, `/stop`, `/mode` | Stream status and controls |
| GET | `/api/investigations/{id}` | Full assessment, timeline, reviews |
| POST | `/api/investigations/{id}/reviews` | Approve or reject a Verify decision with a reason |
| GET | `/api/reviews` | Review history |
| GET | `/api/algorithms`, `/api/algorithms/cluster-graph`, `/api/evaluation` | Model metadata, distributions, training report, cluster graph, evaluation report |

Errors share one shape: `{"error": {"code", "message", "details"}}`. Unknown request fields are
rejected with `422`, so a client cannot send `is_fraud` to influence a decision. A simulation must
use `currency = MDL`, `transfer_type = instant`, one of the nine schemes a customer can start, and
an alias of `+373` plus 8 digits. A personal sender above 5,000 MDL, or a business sender above
100,000 MDL, gets `422`.

`uv run python scripts/api_contract.py --check` calls every route the frontend uses and compares the
JSON shape with `scripts/api_contract.baseline.json`. It fails on a missing key or a type change.
Report fields removed on purpose are listed in the script's `REMOVED` table.
