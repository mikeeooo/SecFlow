# Referință tehnică SecFlow

Pagina descrie pe scurt ce conține proiectul. Explicațiile pe larg sunt în
[DATASET.md](DATASET.md), [ARCHITECTURE.md](ARCHITECTURE.md), [ALGORITHMS.md](ALGORITHMS.md) și
[DESIGN.md](DESIGN.md).

SecFlow evaluează plăți instant din sistemul MIA Plăți Instant al Băncii Naționale a Moldovei.
Pentru fiecare plată, SecFlow calculează un indice de risc și decide **Permit**, **Verify** sau
**Block**. Indicele de risc măsoară cât de neobișnuită este plata. Nu este o probabilitate de
fraudă. Toate datele sunt sintetice.

## Structura codului

| Director | Conținut |
|---|---|
| `storage/` | Doar date: `train.json`, `validation.json`, `test.json` |
| `datagen/` | Generatorul cu seed, identitățile, încărcarea și validarea datelor, `manifest.json` |
| `runtime/` | Indexul de istoric ordonat în timp, dovezile de securitate, starea din memorie, jurnalul de revizii |
| `config/` | `settings.py`, `policy.json` (reguli, ponderi, limite MIA), `limits.py` |
| `ml/` | Caracteristici, algoritmi scikit-learn, politica de decizie, antrenare, evaluare, modelul salvat în `ml/models/` |
| `backend/` | Aplicația FastAPI, serviciul de evaluare, fluxul live simulat, CLI, teste |
| `frontend/` | Interfața React, TypeScript, Vite și Tailwind, cu teste |
| `artifacts/` | Raportul de evaluare: `evaluation.json` și `evaluation.md` |

## Setul de date

Fiecare fișier din `storage/` este un tablou JSON de tranzacții, câte un rând pe linie. Un rând
conține toate câmpurile plății și coloanele de etichetă.

| Fișier | Zile | Plăți evaluate | Fraude | Anomalii legitime | Mărime |
|---|---|---:|---:|---:|---:|
| `train.json` | 1–70 | 70.000 | 1.500 | 3.500 | 79,8 MB |
| `validation.json` | 71–85 | 15.000 | 300 | 750 | 17,1 MB |
| `test.json` | 86–100 | 15.000 | 360 | 750 | 17,1 MB |

Pe lângă plățile evaluate, fișierele conțin încasări primite, marcate cu `"assessed": false`.
Comanda `uv run python -m backend.cli generate` produce fișiere identice octet cu octet la fiecare
rulare, pentru că seed-ul este fix.

### Câmpurile unui rând

Câmpurile sunt grupate după rol.

- **Transfer.** `transaction_id`, `timestamp`, `transfer_type` (mereu `instant`),
  `payment_scheme`, `assessed`, `amount`, `currency` (mereu `MDL`).
- **Plătitor.** `sender_id`, `sender_type`, `sender_name`, `sender_alias`, `sender_home_country`,
  `sender_created_at`.
- **Beneficiar.** `recipient_id`, `recipient_type`, `recipient_name`, `recipient_alias`,
  `recipient_kind`, `recipient_category`, `alias_registered_at`.
- **Sesiune.** `channel`, `device_id`, `session_id`, `auth_method`, `ip_country`,
  `session_events`, `request_to_pay_id`, `qr_payload_merchant_id`.
- **Dovezi de securitate.** `security_evidence`, o listă de dovezi despre sesiune, dispozitiv sau
  părți.
- **Etichete.** `is_fraud`, `is_anomaly`, `fraud_type`, `anomaly_type`, `campaign_id`. Doar
  antrenarea și evaluarea le citesc. Încărcarea pentru aplicație le elimină.

### Reguli MIA aplicate

Limitele sunt în `config/policy.json`, la cheia `mia_limits`. Generatorul nu creează nicio plată pe
care MIA ar refuza-o.

| Segment | Pe tranzacție | Pe zi | Pe lună |
|---|---:|---:|---:|
| Persoană fizică | 5.000 MDL | 10.000 MDL | 30.000 MDL |
| Firmă | 100.000 MDL | 1.000.000 MDL | fără limită |

Limitele pentru firme sunt o presupunere a generatorului. Sursele publice nu dau o valoare oficială.

Schemele de plată sunt `p2p_alias`, `p2p_iban`, `p2b_qr_static`, `p2b_qr_dynamic`, `p2b_link`,
`request_to_pay`, `b2b`, `p2g`, `b2p_salary` și `remittance_in`. Aliasul este un număr de telefon
`+373`, legat de un singur cont.

### Tipuri de fraudă

Zece tipuri apar în toate cele trei fișiere, cu 150, 30 și 30 de plăți:

- `account_takeover_phishing`
- `sim_swap_takeover`
- `vishing_impersonation`
- `request_to_pay_scam`
- `marketplace_scam`
- `investment_scam`
- `romance_scam`
- `money_mule`
- `limit_structuring`
- `coordinated_beneficiary`

Două tipuri apar doar în `test.json`, câte 30 de plăți fiecare: `qr_code_tampering` și
`invoice_redirect`. Cu ele, evaluarea verifică dacă modelul semnalează fraude pe care nu le-a văzut
la antrenare.

### Tipuri de anomalii legitime

Plățile neobișnuite, dar legitime, au unul dintre aceste 14 tipuri:

- `near_limit_planned_purchase`
- `limit_increase_then_large_payment`
- `new_device_legit`
- `travel_abroad`
- `payroll_burst`
- `salary_day_spike`
- `diaspora_remittance_spend`
- `group_payment_new_payee`
- `holiday_spike`
- `night_payment`
- `new_merchant_qr`
- `legit_alias_change`
- `household_repeat_payments`
- `business_settlement`

## Modelul

### Cum se evaluează o plată

Funcția din `ml/assess.py` este singura cale de la o plată la o decizie. O folosesc simulatorul,
fluxul live și evaluarea. Pașii sunt următorii:

1. `build_features` calculează caracteristicile doar din evenimentele anterioare plății.
2. `ModelBundle.score_batch` rulează IsolationForest, LocalOutlierFactor și contextul DBSCAN.
3. `evaluate_rules` aplică regulile și întoarce coduri de motiv.
4. `combine_scores` calculează indicele de risc cu ponderile învățate.
5. `EvidenceStore.active_for` caută dovezi de securitate active la momentul plății.
6. `decide_action` alege Permit, Verify sau Block.

### Caracteristici și reguli

Versiunea caracteristicilor este `features-2.0.0`. Modelul are 31 de caracteristici de plată și 8
caracteristici de context. Șapte dintre caracteristicile de plată sunt specifice MIA:

- `amount_to_tx_cap_ratio`
- `daily_cumulative_to_cap_ratio`
- `near_cap_count_24h`
- `alias_rebound_age_hours`
- `is_request_to_pay`
- `qr_merchant_mismatch`
- `scheme_novelty`

Politica `secflow-evidence-first`, revizia 2.0.0, are 19 reguli. Șase reguli sunt specifice MIA:
`NEAR_TX_CAP`, `DAILY_CAP_PRESSURE`, `CAP_STRUCTURING`, `RECENT_ALIAS_REBOUND`,
`UNKNOWN_RTP_REQUESTER` și `QR_MERCHANT_MISMATCH`.

### Formula și pragurile

Ponderile sunt învățate pe zilele 71–77 din `validation.json`:

```text
risk_score = 100 × (0.65 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.30 × rules)
```

Pragurile sunt alese pe zilele 78–85 din `validation.json`.

| Segment | Verify de la | Block de la |
|---|---:|---:|
| Persoană fizică | 71 | 91 |
| Firmă | 69 | 91 |

Block pe baza scorului cere cel puțin două familii de reguli diferite. O dovadă de securitate de
încredere și activă blochează plata indiferent de scor. Dovezile expirate, retrase, neconfirmate
sau din surse nesigure nu blochează.

## Rezultate pe `test.json`

Rezultatele sunt măsurate pe date sintetice. Nu dovedesc cât de bine ar detecta SecFlow fraude
reale.

| Metrică | Valoare |
|---|---|
| Fraude trimise la Verify sau Block | 76,7% (276 din 360) |
| Fraude blocate | 30,8% (111 din 360) |
| Precizia blocărilor | 96,5% (111 din 115). Obiectivul de 99% nu este atins. |
| Rata Verify | 4,58%. Obiectivul de cel mult 5% este atins. |
| Blocări false pentru plăți legitime | 0,03% |

Detecția este slabă pentru `investment_scam` și `romance_scam`, cu 36,7% fiecare. Pentru
`invoice_redirect` este 50%. Tabelul complet pe tipuri de fraudă și de anomalie este în
[raportul de evaluare](../artifacts/evaluation.md). Comanda `uv run python -m backend.cli evaluate`
îl regenerează.

## API

Backend-ul expune aceste rute:

| Metodă | Rută | Rol |
|---|---|---|
| GET | `/api/health` | Starea serviciului |
| GET | `/api/overview`, `/api/dashboard` | Datele pentru panouri |
| GET | `/api/transactions` | Plățile evaluate, cu filtre și paginare |
| GET | `/api/accounts`, `/api/accounts/{account_id}` | Conturile și detaliul unui cont |
| GET | `/api/parties` | Conturile și contrapărțile pentru căutare |
| GET | `/api/simulator/options` | Opțiunile simulatorului |
| POST | `/api/simulations` | Evaluează o plată simulată |
| GET | `/api/live-transfers` | Fluxul live |
| GET | `/api/simulation-stream` | Starea fluxului live |
| POST | `/api/simulation-stream/start`, `/stop`, `/mode` | Pornește, oprește sau schimbă modul fluxului live |
| GET | `/api/investigations/{request_id}` | Detaliul unei decizii |
| POST | `/api/investigations/{request_id}/reviews` | Revizia umană a unei plăți Verify |
| GET | `/api/reviews` | Istoricul reviziilor |
| GET | `/api/algorithms`, `/api/algorithms/cluster-graph` | Detaliile algoritmilor |
| GET | `/api/evaluation` | Raportul de evaluare |

`POST /api/simulations` întoarce 422 în trei cazuri:

- Moneda nu este `MDL`.
- Tipul transferului nu este `instant`.
- O persoană fizică trimite mai mult de 5.000 MDL.

## Interfața

Interfața are aceste pagini: **Overview** (`/`), **Live Transfers** (`/live`), **Transactions**
(`/transactions`), **Accounts** (`/accounts`), **Investigation** (`/investigations/:requestId`),
**Review History** (`/history`), **Simulator** (`/simulator`) și **Algorithms** (`/algorithms`).
Sumele sunt afișate în MDL. Simulatorul are un selector de schemă de plată.

## Comenzi

Rulează comenzile din rădăcina proiectului.

| Comandă | Efect |
|---|---|
| `uv run python -m backend.cli generate` | Scrie cele trei fișiere din `storage/` și `datagen/manifest.json` |
| `uv run python -m backend.cli validate` | Verifică structura, numărul de rânduri, limitele MIA, partițiile, hash-urile și reproductibilitatea |
| `uv run python -m backend.cli train` | Antrenează pe `train.json` și alege ponderile și pragurile pe `validation.json` |
| `uv run python -m backend.cli evaluate` | Evaluează o singură dată pe `test.json` |
| `uv run uvicorn backend.api:app --host 127.0.0.1 --port 8001` | Pornește backend-ul |
| `npm --prefix frontend run dev` | Pornește interfața pe <http://localhost:5173> |
| `uv run pytest` | Rulează testele backend |
| `npm --prefix frontend test -- --run` | Rulează testele frontend |

Pe Vercel, `vercel.json` publică backend-ul și interfața ca două servicii în același proiect.

## Limitări

- Aplicația încarcă doar `test.json`. În primele zile de test, plățile nu au istoric anterior în
  aplicație. Scorurile din aplicație pot diferi de cele din evaluare.
- Simulările și fluxul live există doar în memoria procesului. O repornire le șterge.
- Istoricul reviziilor este scris în `runtime/state/review_history.sqlite3`. Rulează un singur
  proces backend, pentru că fiecare proces are propria stare.
- Aplicația nu are autentificare. Numele revizorului este doar text afișat.
