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
| `config/` | `settings.py` (variabilele `SECFLOW_*`, listate în `.env.example`), `policy.json` (ponderi, reguli, obiective, limite MIA), `limits.py` |
| `ml/` | Istoricul ordonat în timp, caracteristici, cei trei algoritmi scikit-learn, reguli și decizie, dovezi de securitate, antrenare, evaluare, modelul salvat în `ml/models/` |
| `scripts/` | `api_contract.py`, verificarea formei răspunsurilor API de care depinde interfața |
| `backend/` | Aplicația FastAPI, serviciul de evaluare, fluxul live simulat, CLI, teste |
| `frontend/` | Interfața React, TypeScript, Vite și Tailwind, cu teste |
| `artifacts/` | Raportul de evaluare: `evaluation.json` și `evaluation.md`. Ora evaluării și latența sunt în `timings.json`. |

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

`Assessor.assess_batch` din `ml/assess.py` este singura cale de la o plată la o decizie. O folosesc
aplicația la pornire, simulatorul, fluxul live și evaluarea. Pașii sunt următorii:

1. `build_features` calculează caracteristicile doar din evenimentele anterioare plății.
2. `ModelBundle.score_batch` rulează IsolationForest, LocalOutlierFactor și contextul DBSCAN.
3. `evaluate_rules` aplică regulile și întoarce coduri de motiv.
4. `combine_scores` calculează indicele de risc cu ponderile fixe din `config/policy.json`.
5. `EvidenceStore.active_for` caută dovezi de securitate active la momentul plății.
6. `decide_action` alege Permit, Verify sau Block.

### Antrenarea

Cei trei algoritmi nu folosesc etichete. Toți sunt antrenați doar pe `train.json`, cu parametri
ficși:

| Algoritm | Parametri |
|---|---|
| IsolationForest | 200 de arbori, 256 de plăți pe arbore |
| LocalOutlierFactor (`novelty=True`) | 20 de vecini, 3 caracteristici de sumă, un eșantion de 20.000 de plăți |
| DBSCAN | `min_samples = 5`, `eps` = percentila 90 a distanței până la al 5-lea vecin (0,705) |

Scorurile brute IF și LOF sunt transformate în procente prin distribuția lor pe cele 15.000 de plăți
din `validation.json`, fără etichete. Etichetele din `validation.json` servesc doar la alegerea
pragurilor. Etichetele din `test.json` servesc doar la evaluarea finală. Antrenarea durează în jur
de 5 secunde.

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

Politica `secflow-evidence-first`, revizia 3.0.0, are 19 reguli. Șase reguli sunt specifice MIA:
`NEAR_TX_CAP`, `DAILY_CAP_PRESSURE`, `CAP_STRUCTURING`, `RECENT_ALIAS_REBOUND`,
`UNKNOWN_RTP_REQUESTER` și `QR_MERCHANT_MISMATCH`.

### Formula și pragurile

Ponderile sunt fixe, în `config/policy.json`:

```text
risk_score = 100 × (0.65 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.30 × rules)
```

Pragurile sunt alese printr-o căutare pe grilă pe `validation.json`. Căutarea păstrează perechile cu
cel mult 4% plăți trimise la Verify, cel mult 0,1% plăți legitime blocate și cel puțin 99% fraude
printre blocări, apoi o alege pe cea care prinde cele mai multe fraude. Obiectivul Verify este 5%,
dar un punct procentual rămâne liber pentru conturile noi, care lipsesc din validare. Pragurile alese
sunt **Verify de la 68** și **Block de la 86**, aceleași pentru persoane fizice și firme.

Block pe baza scorului cere cel puțin două familii de reguli diferite. O dovadă de securitate de
încredere și activă blochează plata indiferent de scor. Dovezile expirate, retrase, neconfirmate
sau din surse nesigure nu blochează.

## Rezultate pe `test.json`

Rezultatele sunt măsurate pe date sintetice. Nu dovedesc cât de bine ar detecta SecFlow fraude
reale.

| Metrică | Valoare |
|---|---|
| Fraude trimise la Verify sau Block | 78,6% (283 din 360) |
| Fraude blocate | 34,7% (125 din 360) |
| Precizia blocărilor | 96,9% (125 din 129). Obiectivul de 99% nu este atins. |
| Rata Verify | 4,49%. Obiectivul de cel mult 5% este atins. |
| Blocări false pentru plăți legitime | 0,03% (4 din 14.640). Obiectivul de cel mult 0,1% este atins. |

Regulile singure egalează scorul combinat. Pe test, regulile au AP 0,675 [0,628; 0,721], iar
scorul combinat are 0,668 [0,621; 0,714]. Diferența pereche este -0,007 [-0,023; 0,009], deci
detectorii de anomalii nu adaugă o AP măsurabilă. Decizia de a păstra sistemul și criteriul nou sunt
în [ADR 0003](adr/0003-fixed-weights-and-the-rules-only-result.md).

Detecția este slabă pentru `romance_scam` (40%), `investment_scam` (50%) și `invoice_redirect`
(33,3%). Tabelul complet pe tipuri de fraudă și de anomalie este în
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
| `uv run python -m backend.cli validate` | Verifică fișierele, hash-urile, numărul de plăți, limitele MIA, tipurile de fraudă și eliminarea etichetelor |
| `uv run python -m backend.cli train` | Antrenează pe `train.json` și alege pragurile pe `validation.json` |
| `uv run python -m backend.cli evaluate` | Evaluează o singură dată pe `test.json` |
| `uv run uvicorn backend.api:app --host 127.0.0.1 --port 8001` | Pornește backend-ul |
| `npm --prefix frontend run dev` | Pornește interfața pe <http://localhost:5173> |
| `uv run pytest --cov` | Rulează testele backend, cu pragul de acoperire de 97% |
| `uv run mypy` | Verifică tipurile în modul strict |
| `uv run lint-imports` | Verifică straturile `backend -> ml -> datagen -> config` |
| `uv run python scripts/reproduce.py` | Regenerează datele, modelul și evaluarea și le compară cu fișierele salvate |
| `uv run python scripts/experiments.py --check` | Compară scorul combinat cu fiecare componentă |
| `npm --prefix frontend run lint` | Rulează oxlint pe interfață |
| `npm --prefix frontend test -- --run` | Rulează testele frontend |
| `uv run python scripts/api_contract.py --check` | Verifică forma răspunsurilor API față de `scripts/api_contract.baseline.json` |

Pe Vercel, `vercel.json` publică backend-ul și interfața ca două servicii în același proiect.

## Limitări

- Aplicația încarcă `validation.json` doar ca istoric și afișează plățile din `test.json`. Vede
  zilele 71–100, iar evaluarea vede zilele 1–100. Rata Verify este circa 5% în aplicație și 4,49% în
  raport.
- Simulările, fluxul live și reviziile există doar în memoria procesului. O repornire le șterge.
  Rulează un singur proces backend, pentru că fiecare proces are propria stare.
- Aplicația nu are autentificare. Numele revizorului este doar text afișat.
- Testul nu este complet neatins. O primă rulare a trimis 5,81% din plățile de test la Verify. Rezerva
  pentru conturile noi și ponderile a cinci reguli au fost alese apoi pe cifre din validare, dar
  motivul căutării a venit din rezultatul pe test. Detalii în [ALGORITHMS.md](ALGORITHMS.md).
