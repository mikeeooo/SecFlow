# Dataset: synthetic MIA Plăți Instant payments, Kaggle-style partitions

The dataset is **fully synthetic**. `datagen/generator.py` (`secflow-mia-generator-2.0.0`, schema 2.0,
seed 42) produces it, modelled on **MIA Plăți Instant**, the instant-payment system of the National
Bank of Moldova (BNM). It represents no real person, business, phone number, payment or fraud case.
Names are generated combinations of common Moldovan first and last names. Phone aliases are random
numbers in the MIA format (`+373` and 8 digits). Business names combine generic place words with a
trade. The public offices are labelled "(simulated)".

## Three files, nothing else

`storage/` contains exactly three files, like a Kaggle competition:

| File | Days (dates) | Rows | Assessed payments | Incoming credits | Fraud | Unusual legitimate | Size |
|---|---|---:|---:|---:|---:|---:|---:|
| `storage/train.json` | 1–70 (5 Jan – 15 Mar 2026) | 75,416 | 70,000 | 5,416 | 1,500 | 3,500 | 79.8 MB |
| `storage/validation.json` | 71–85 (16 – 30 Mar 2026) | 16,138 | 15,000 | 1,138 | 300 | 750 | 17.1 MB |
| `storage/test.json` | 86–100 (31 Mar – 14 Apr 2026) | 16,205 | 15,000 | 1,205 | 360 | 750 | 17.1 MB |

Each file is one JSON array of self-contained transaction rows. Rows are written compactly, one per
line, so the files stay valid JSON and diff cleanly. Indented with two spaces, `train.json` would be
103 MB, which is over GitHub's 100 MB file limit. `validate` fails if `storage/` holds anything
else. Dotfiles such as `.DS_Store` are ignored.

Generation metadata (seed, versions, counts per partition and per type, SHA-256 and size per file,
cold-start account IDs, drift biller IDs, evidence summary, limits) is in `datagen/manifest.json`, on
the code side, never in `storage/`.

| File | Used for |
|---|---|
| `train.json` | Fitting preprocessing, IsolationForest, LOF and DBSCAN. Labels are not used for fitting. |
| `validation.json` | The reference scores for the IsolationForest and LOF score mapping (no labels), then the Verify/Block threshold search (labels). See `ml/training.py`. |
| `test.json` | One frozen final evaluation (`ml/evaluation.py`). It is also the history the app serves, which the simulator and the live stream continue from. |

There is no separate demonstration dataset. The service loads only `test.json` (its runtime view),
so its features see the 15 test days and nothing earlier. The evaluation loads all three files, so
test features also see days 1–85 as history. A test payment can therefore score differently in the
app than in the evaluation, mostly in the first test days, where the app has little history.

## MIA facts used

- Real time, 24/7, credited in under 15 seconds, final (no recall). The runtime ledger settles
  every assessed payment 5 seconds after its timestamp.
- Currency **MDL** only: every row has `currency = "MDL"`. The loader and the API reject anything else.
- Identifier: the phone-number alias (`+373` + 8 digits), one alias per account at a time. IBAN
  transfers also exist.
- Limits (`config/policy.json`, `mia_limits`): **5,000 MDL per transaction, 10,000 MDL per day and
  30,000 MDL per month for personal accounts** (bank level). Business payments (P2B/B2B) are not bound
  by the P2P cap. The generator assumes 100,000 MDL per transaction and 1,000,000 MDL per day for
  businesses. The generator never creates a payment MIA would refuse: a personal customer who has
  used today's or this month's limit simply makes no further payment. Days and months are UTC.
- Fraud pressure in Moldova (BNM warnings, Eurojust 2025): vishing with spoofed bank or police
  numbers and cloned voices, phishing with the BNM brand, Request-to-Pay disguised as "receive
  money", and call-centre scams laundered through mule accounts.

Sources: BNM, MIA Plăți Instant (mia.bnm.md); BRD guide to MIA limits; MICB MIA terms; BNM questions
and answers about instant payments; BNM news on business and QR payments; Radio Moldova report of a
BNM fraud warning; Colenco legal blog on the RTP scam vector; Eurojust press release (2025). Full
links are in `docs/superpowers/specs/2026-11-27-mia-kaggle-dataset-design.md`.

## Row schema

| Group | Columns |
|---|---|
| Identification and transfer | `transaction_id` (`TX-0000001`…, global chronological order), `timestamp` (ISO-8601 UTC `Z`), `transfer_type` (always `instant`), `payment_scheme`, `assessed`, `amount` (MDL, 2 decimals), `currency` (`MDL`) |
| Sender | `sender_id`, `sender_type` (`personal` \| `business` \| `external`), `sender_name`, `sender_alias` (phone alias or null), `sender_home_country`, `sender_created_at` (null for external payers) |
| Recipient | `recipient_id`, `recipient_type` (`account` \| `counterparty`), `recipient_name`, `recipient_alias`, `recipient_kind` (`person` \| `business` \| `public_institution`), `recipient_category`, `alias_registered_at` (when the recipient's alias was last bound) |
| Session context | `channel` (`mobile_app` \| `web`), `device_id`, `session_id`, `auth_method`, `ip_country`, `session_events`, `request_to_pay_id`, `qr_payload_merchant_id` |
| Trusted evidence | `security_evidence` |
| **Labels (offline only)** | `is_fraud`, `is_anomaly`, `fraud_type`, `anomaly_type`, `campaign_id` |

- `assessed = true`: an outgoing payment initiated by a SecFlow account; it is scored.
  `assessed = false`: a context-only incoming credit from an external payer (salary, client payment,
  diaspora remittance, mule inflow). Its device, session and sign-in columns are null.
- `session_events`: this session's authentication events before the payment, each
  `{event_id, timestamp, event_type, auth_method, ip_country, device_id, reason}`. The event types
  are `login_failure`, `login_success`, `session_revoked` (administrative logout),
  `alias_rebound` (the phone alias moved to a new SIM or device) and `limit_changed`.
- `request_to_pay_id` is set only for `request_to_pay` payments. `qr_payload_merchant_id` is set only
  for QR schemes. It names the merchant the QR claims, and equals `recipient_id` unless the code was
  tampered with.
- `security_evidence`: the trusted-source evidence records about this row's session, device, sender
  or recipient, each `{evidence_id, evidence_type, subject_type, subject_id, status, source,
  source_trust, recorded_at, effective_at, expires_at}`. A record can be recorded after the
  payment. The runtime evidence store is time-aware, so such a record never applies to that payment
  (see below).
- Labels: `is_fraud`, `fraud_type` (null if legitimate), `anomaly_type` (null if routine),
  `is_anomaly` (true for every anomaly row and for fraud that looks anomalous), `campaign_id`
  (fraud campaign `CMP-…` or unusual episode `EP-…`). On incoming credits the labels are false or
  null; `campaign_id` links a mule campaign's inflows.

### Payment schemes

`p2p_alias`, `p2p_iban`, `p2b_qr_static`, `p2b_qr_dynamic`, `p2b_link`, `request_to_pay`, `b2b`, `p2g`
(public institutions), `b2p_salary` (payroll, assessed when a SecFlow business pays its staff, context
when salary arrives) and `remittance_in` (incoming diaspora transfers, never assessed). Training file
row counts by scheme: b2b 32,989; p2p_alias 10,565; p2b_qr_static 8,024; p2b_qr_dynamic 6,878;
p2b_link 5,853; p2p_iban 4,770; request_to_pay 2,530; b2p_salary 2,155; p2g 1,399; remittance_in 253.

## The runtime view

`datagen.dataset.load_partition(path)` and `load_partitions(paths)` read the rows, **drop the label
columns as each row is read**, and rebuild everything the pipeline needs:

- accounts from the sender columns of assessed rows;
- counterparties from the recipient columns and the external payers;
- payment requests from assessed rows;
- the ledger (a `transfer_settled` event 5 s after each assessed payment, an `incoming_credit` for
  each context row);
- authentication events from `session_events`, deduplicated by `event_id`;
- the evidence stream from `security_evidence`, deduplicated by `evidence_id`.

Rows without `transfer_type = "instant"` or with a currency other than MDL are refused
(`DatasetError`). Labels have a separate loader (`load_labels`) that only training and evaluation
call. The service never calls it. A test changes every label in a copy of `test.json` and checks
that the runtime view stays identical.

## Population

400 accounts (336 personal, 64 business, all resident in Moldova) are shared by all partitions. 40
personal **cold-start accounts** are opened during days 86–97 and make their first payment in
`test.json`. There are 650 base counterparties: 25 popular billers, 8 simulated public offices, merchants and
persons. There are also 90 diaspora remitters abroad. Payees nobody has paid before keep appearing
over time.

Routine behaviour per account is generator-only (never written): activity rate, typical amount
(median personal payment 156 MDL, median business payment 5,381 MDL in `train.json`), weighted
favourite payees, one or two devices, a sign-in mix, active hours, two paydays a month (advance on
the 5th–12th, balance on the 24th–28th) and, for a quarter of personal accounts, a monthly remittance
from a relative abroad. Routine traffic also pays brand-new payees (6%). It includes failed sign-ins,
foreign sessions, permanent device replacements, legitimate Request-to-Pay from billers and friends,
and QR payments at merchants.

## Fraud taxonomy (`fraud_type`)

Each of the ten types below has **150 payments in train, 30 in validation and 30 in test**.

| Type | What happens |
|---|---|
| `account_takeover_phishing` | Credentials phished on a fake bank or BNM page. New device, failed sign-ins, often a foreign or VPN session, sometimes a raised limit; payments near the 5,000 MDL cap to new payees. 20% are subtle. |
| `sim_swap_takeover` | `alias_rebound` plus a new device shortly before; payments drain toward the 10,000 MDL daily limit. 15% are subtle. |
| `vishing_impersonation` | Fake bank or police call. The victim's own device and strong sign-in, often at unusual hours; one to three near-cap payments to a new "safe account". 40% look ordinary. |
| `request_to_pay_scam` | Victims approve a Request-to-Pay from an unknown requester disguised as "receive money" (three to six victims per requester). 60% look ordinary. |
| `marketplace_scam` | A fake classifieds seller; several buyers pay a mid-size amount within a few days. |
| `investment_scam` | Escalating payments to the same new "platform" payee over days or weeks. |
| `romance_scam` | Recurring moderate payments to one new person over weeks. 80% look ordinary. |
| `money_mule` | Several incoming credits, then rapid pass-through to one or two new beneficiaries, often with recently registered aliases. |
| `limit_structuring` | A large sum split into payments just under the per-transaction cap (≥ 90% of 5,000 MDL), at most two a day so the daily limit holds, across two or three payees. |
| `coordinated_beneficiary` | Several compromised accounts pay one new beneficiary within hours. |

The last two types appear **only in `test.json`**, 30 payments each, as a novelty check:

| Type | What happens |
|---|---|
| `qr_code_tampering` | A static merchant QR is replaced. Customers pay a new account while `qr_payload_merchant_id` still names the real merchant. |
| `invoice_redirect` | Business email compromise. A business pays a brand-new fake supplier (`b2b`) an amount close to a usual invoice, from a known device with strong sign-in. |

Every campaign stays inside one partition (multi-day campaigns start early enough to finish before
the boundary).

## Anomaly taxonomy (`anomaly_type`, unusual but legitimate, 5%)

`near_limit_planned_purchase`, `limit_increase_then_large_payment`, `new_device_legit`,
`travel_abroad`, `payroll_burst`, `salary_day_spike`, `diaspora_remittance_spend`,
`group_payment_new_payee`, `holiday_spike` (Orthodox Christmas on 7 January, Mărțișor on 1 March,
8 March, and Orthodox Easter on 12 April, each with the two days before; 1 September is outside the
date range, and validation has no holiday), `night_payment`, `new_merchant_qr`, `legit_alias_change`
(a customer moves to a new phone number), `household_repeat_payments` and `business_settlement`. All
14 appear in train. Validation and test contain every type except `holiday_spike` in validation.

## Drift in the test period

Routine amounts are 12% higher from day 86. Three new billers (`CP-00651`–`CP-00653`) are first paid
in the test period: 30% of accounts adopt one from a random day between 86 and 100.

## No single field identifies fraud

Measured on `train.json` (assessed payments):

- Fraud uses every sign-in method legitimate traffic uses (biometric 600, password_otp 433,
  password 313, otp 154 of 1,500 fraud payments).
- 542 of 1,500 fraud payments are labelled `is_anomaly = false`.
- 2,355 legitimate payments follow failed sign-ins in their session.
- 126 legitimate payments follow an `alias_rebound` (150 fraud payments do).
- 84 legitimate payments follow a `limit_changed` event.
- 2,380 legitimate payments are Request-to-Pay (150 fraud payments are).
- 124 legitimate personal payments are ≥ 4,500 MDL, close to the cap (397 fraud payments are).
- 1,322 legitimate payments come from a session outside Moldova (70 fraud payments do).
- A QR merchant mismatch occurs only in the test-only `qr_code_tampering` type, so training never
  sees it.

## Simulated security evidence

Evidence models a trusted server-side security source: sessions confirmed compromised, devices
confirmed in a malware investigation, and beneficiaries confirmed to be controlled by a fraudulent
operation. It covers 157 campaign records for a documented subset of campaigns: account takeover
35% (session, after the first payment), SIM swap 25% (device), vishing 15%, RTP scam 30%,
marketplace 25%, investment 15%, romance 10%, money mule 30%, coordinated beneficiary 40%, QR
tampering 20% (beneficiary). Structuring and invoice redirection get none. Most fraud has no
evidence: in the test file, 24 of 360 fraud payments had active evidence at payment time.

The store treats a record as active at time *t* only if it was recorded at or before *t*,
`effective_at ≤ t < expires_at`, its status is `confirmed`, its source is trusted, and its type
blocks. There are ten **never-block fixtures** (`SEC-FX-*`), two each of expired, retracted,
future-recorded, pending-review and untrusted-source evidence. All of them point at popular billers
that legitimate customers keep paying. The evidence tests cover each of these five cases. There are
also 25 administrative session revocations, which are not fraud evidence. All evidence is
simulated; none of it is a real watchlist.

## Validation and reproducibility

```bash
uv run python -m backend.cli generate   # writes storage/*.json and datagen/manifest.json, then validates
uv run python -m backend.cli validate
uv run pytest backend/tests/test_dataset.py
```

`validate` (`datagen/dataset.py`) checks these points:

- `storage/` holds exactly the three files;
- the SHA-256 of each file matches the manifest;
- each partition has 1,000 assessed payments a day;
- every row has `transfer_type = instant` and `currency = MDL`;
- no personal payment is above 5,000 MDL, and no personal sender-day or sender-month is above the
  daily or monthly limit;
- every fraud type has its expected count per partition, and the test-only types are absent from
  train and validation;
- the runtime loader accepts each file and drops the label columns;
- no row appears in two files (row fingerprints).

`backend/tests/test_dataset.py` also runs the generator again from the seed into a temporary
directory and checks that every file matches the committed one byte for byte.
