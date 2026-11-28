# MIA instant-payment dataset, Kaggle-style partitions — design

Date: 2026-11-27. Branch: `feat/mia-dataset`. Supersedes the dataset parts of
`2026-11-26-algorithm-dataset-improvements-design.md`.

## Goal

1. `storage/` contains **only data**: three Kaggle-style JSON files, `storage/train.json`,
   `storage/validation.json`, `storage/test.json`. No Python, no manifests, no other files there.
2. The synthetic dataset models **MIA Plăți Instant**, the instant-payment system of the National
   Bank of Moldova (BNM), and covers the fraud and anomaly types that occur on that kind of rail.
3. Every transaction keeps the mandatory field `"transfer_type": "instant"`.

## Decisions (taken by the user)

| Question | Decision |
|---|---|
| Where the Python modules in `storage/` go | New packages **`datagen/`** (generator, identities, dataset loader/validator, records) and **`runtime/`** (history, evidence, memory, review_log). All imports updated. |
| 4 partitions + demo → 3 parts | **Calibration merged into validation; demo removed; test also serves as the app's history** for the live stream and simulator. |
| File format | **One JSON file per part**: a JSON array of transaction rows with all fields and label columns, like a Kaggle competition. |

Generated metadata that used to sit in `manifest.json` (seeds, versions, counts, SHA-256, cold-start
IDs) moves to `datagen/manifest.json` (code side, not in `storage/`). Review-history state under
`storage/state/` moves to `runtime/state/` (still git-ignored). `config/settings.py` exposes
`train_path`, `validation_path`, `test_path`.

## MIA domain facts used (sources at the end)

- Real time, 24/7, credited in < 15 s, payments are final (no recall).
- Currency **MDL** only. `currency` becomes `"MDL"`; the API rejects any other.
- Identifier: the **phone-number alias** (`+373` + 8 digits); one alias maps to one account at a
  time. IBAN transfers also exist.
- Limits for individuals (BNM / bank terms): **5,000 MDL per transaction**, **10,000 MDL per day**,
  **30,000 MDL per month** (bank level; BNM recommended up to 25,000/day). Fee-free up to
  10,000 MDL/month. Business payments (P2B/B2B) are not bound by the P2P cap; the generator uses
  100,000 MDL per transaction for them (documented assumption). Limits live in
  `config/policy.json` under `mia_limits`.
- Payment schemes (`payment_scheme`): `p2p_alias`, `p2p_iban`, `p2b_qr_static`,
  `p2b_qr_dynamic`, `p2b_link`, `request_to_pay`, `b2b`, `p2g` (public institutions),
  `b2p_salary` (incoming payroll), `remittance_in` (incoming, diaspora).
- Fraud pressure in Moldova (BNM warnings, Eurojust 2025): vishing with spoofed bank/police numbers
  and cloned voices, phishing with the BNM brand, Request-to-Pay disguised as "receive money",
  call-centre scams laundered through mule accounts.

## Row schema (every row in every file)

Identification and transfer: `transaction_id`, `timestamp` (ISO-8601 UTC `Z`), `transfer_type`
(always `instant`), `payment_scheme`, `assessed` (true for outgoing payments initiated by a SecFlow
account, which are scored; false for context-only incoming credits from external payers),
`amount` (MDL, 2 decimals), `currency` (`MDL`).

Sender: `sender_id`, `sender_type` (`personal`|`business`|`external`), `sender_name`,
`sender_alias` (phone alias or null), `sender_home_country`, `sender_created_at`.

Recipient: `recipient_id`, `recipient_type` (`account`|`counterparty`), `recipient_name`,
`recipient_alias`, `recipient_kind`, `recipient_category`, `alias_registered_at` (when the
recipient's alias was last bound; supports SIM-swap / alias re-registration signals).

Session context: `channel` (`mobile_app`|`web`), `device_id`, `session_id`, `auth_method`,
`ip_country`, `session_events` (nested list of this session's auth events before the payment:
`{event_id, timestamp, event_type, auth_method, ip_country, device_id, reason}`; event types
`login_failure`, `login_success`, `session_revoked`, `alias_rebound`, `limit_changed`),
`request_to_pay_id` (null unless the payment answers an RTP), `qr_payload_merchant_id` (QR schemes:
the merchant the QR claims; differs from `recipient_id` when tampered).

Trusted security evidence known at that moment: `security_evidence` (list of
`{evidence_id, subject_type, subject_id, status, source, source_trust, recorded_at, effective_at,
expires_at}`), including the never-block fixtures (expired, retracted, future-recorded, pending,
untrusted).

Labels (Kaggle target columns; **offline only**, dropped by the runtime loader): `is_fraud`,
`is_anomaly`, `fraud_type` (null if legitimate), `anomaly_type` (null if routine), `campaign_id`.

Accounts, counterparties, auth events, incoming credits and evidence are rebuilt from the rows by
`datagen.dataset.load_partition` (runtime view, labels dropped). There are no other files.

## Partitions

100 days, 1,000 assessed payments per day (plus incoming credits), chronological, whole days:

| File | Days | Use |
|---|---|---|
| `train.json` | 1–70 | Fit preprocessing, IF, LOF, DBSCAN (labels unused for fitting) |
| `validation.json` | 71–85 | IsolationForest and LOF score references (no labels), then the Verify/Block threshold search (labels). See `ml/training.py`. |
| `test.json` | 86–100 | One frozen evaluation; also the history the app serves and the live stream continues from |

Campaigns never cross a partition boundary. 40 cold-start accounts are created in the test period.
Drift (new billers, +12% amounts) applies to the test period.

## Fraud taxonomy (`fraud_type`)

Present in all three partitions (about 150 train / 30 validation / 30 test rows each):

1. `account_takeover_phishing`: credentials phished (fake bank/BNM page); new device, failed
   logins, foreign or VPN `ip_country`, payments near the 5,000 MDL cap to new payees.
2. `sim_swap_takeover`: `alias_rebound` plus a new device shortly before payments; drains toward the
   daily limit.
3. `vishing_impersonation`: fake bank/police call (spoofed number). Victim's own known device and
   strong auth, urgency, one or more near-cap payments to a new "safe account" payee, often at
   unusual hours.
4. `request_to_pay_scam`: victim approves an RTP from an unknown requester disguised as "receive
   money"; `payment_scheme = request_to_pay`, new requester, many victims per requester.
5. `marketplace_scam`: fake seller on classifieds; victims pay a new personal payee a mid amount,
   several victims per payee within days.
6. `investment_scam`: escalating repeated payments to the same new payee over days or weeks.
7. `romance_scam`: recurring moderate payments to one new payee over weeks.
8. `money_mule`: several incoming credits, then rapid pass-through to 1–2 new beneficiaries
   (including recently opened accounts).
9. `limit_structuring`: a large sum split into many payments just under the per-transaction cap,
   staying under the daily limit, across several payees.
10. `coordinated_beneficiary`: several compromised accounts pay one new beneficiary within hours.

Unseen in train and validation, **test only** (novelty check):

11. `qr_code_tampering`: a static merchant QR is replaced; customers pay a new account while
    `qr_payload_merchant_id` names the real merchant.
12. `invoice_redirect`: business email compromise; a business pays a new fake supplier an amount
    close to a usual invoice (`b2b`).

## Anomaly taxonomy (`anomaly_type`, legitimate but unusual, about 5%)

`near_limit_planned_purchase`, `limit_increase_then_large_payment`, `new_device_legit`,
`travel_abroad`, `payroll_burst`, `salary_day_spike`, `diaspora_remittance_spend`,
`group_payment_new_payee`, `holiday_spike` (Orthodox Christmas/Easter, Mărțișor, 1 September
inside the date range), `night_payment`, `new_merchant_qr`, `legit_alias_change` (new phone
number), `household_repeat_payments`, `business_settlement`.

`is_anomaly` is true for every `anomaly_type` row and for fraud rows that look anomalous. A share of
fraud rows stays `is_anomaly = false` (vishing, RTP and romance scams look ordinary).

## Model and rules impact

- Features keep the same family, recomputed in MDL, plus MIA features: `amount_to_tx_cap_ratio`,
  `daily_cumulative_to_cap_ratio`, `near_cap_count_24h`, `alias_rebound_age_hours`,
  `is_request_to_pay`, `qr_merchant_mismatch`, `scheme_novelty` (first use of this scheme by the
  sender). `FEATURE_VERSION` becomes `features-2.0.0`.
- New rules in `config/policy.json` (revision 2.0.0): `NEAR_TX_CAP`, `DAILY_CAP_PRESSURE`,
  `CAP_STRUCTURING`, `RECENT_ALIAS_REBOUND`, `UNKNOWN_RTP_REQUESTER`, `QR_MERCHANT_MISMATCH`, in
  existing or new corroboration families (`limit_pressure`, `identity_change`,
  `payment_instrument`). Amount parameters converted to MDL.
- Retrain and re-evaluate; update `ml/models/*`, `artifacts/evaluation.*` and the README metrics.
  Report recall per `fraud_type`, including the two test-only types.

## App impact

- Backend: the service loads `test.json` as history. Simulator payload: `currency: Literal["MDL"]`,
  `payment_scheme`; for personal senders `amount` above 5,000 is rejected (422); optional
  `recipient_alias`. The stream producer emits MIA schemes.
- Frontend: money shown in MDL, scheme shown in rows and details, scheme selector in the simulator.
  Types and tests updated.

## Validation and tests

`python -m backend.cli generate|validate|train|evaluate` keep working. `validate` checks: exactly
three files in `storage/` and nothing else; row counts; unique IDs; chronology and whole-day
partitions; every row has `transfer_type = instant` and `currency = MDL`; no personal payment above
the per-transaction cap (MIA enforces it, so structuring stays under it); every `fraud_type` present
with expected counts; test-only types absent from train and validation; campaigns within one
partition; no label columns in the runtime view; zero duplicate fingerprints across files;
reproducibility from the seed (byte-identical files).

## Sources

- BNM, MIA Plăți Instant: https://mia.bnm.md/ro
- BRD, MIA limits: https://brd.gov.md/revenire/ghid-pentru-suport-la-revenire/informatii-utile/sistemul-de-plati-instant-mia/
- MICB terms: https://micb.md/services/mia-plati-instant/
- BNM Q&A: https://www.bnm.md/en/content/questions-and-answers-about-instant-payments
- BNM business and QR expansion: https://bnm.md/en/print/content/more-1-million-users-mia-instant-payments-expands-options-businesses-b2b-payments-qr-code
- Radio Moldova, BNM fraud warning: https://radiomoldova.md/p/77562/moldova-s-central-bank-warns-of-sophisticated-digital-fraud
- Colenco, RTP scam vector: https://colenco.legal/en/blog/financial-safety-moldova-cash-limits-mia
- Eurojust 2025 operation: https://www.eurojust.europa.eu/news/eurojust-assists-operation-romania-and-moldova-against-laundering-phishing-fraud-proceeds
