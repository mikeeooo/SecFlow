# SecFlow

SecFlow analizează plăți instant **MIA Plăți Instant** (sistemul de plăți instant al Băncii Naționale
a Moldovei, în lei moldovenești, MDL) și explică ce decide. Fiecare plată primește un indice de risc
de anomalie construit din trei algoritmi scikit-learn (IsolationForest, LocalOutlierFactor, DBSCAN)
și din reguli explicite de comportament. O politică de tip „mai întâi dovezile” transformă apoi
acest indice în **Permit** (permite), **Verify** (verifică) sau **Block** (blochează), iar fiecare
decizie își arată calculul, codurile de motiv și sursa acțiunii.

```text
risk_score = 100 × (0.65 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.30 × rules)
```

Ponderile sunt învățate pe zilele 71–77 din `validation.json` (căutare exhaustivă pe simplex, cu LOF
menținut la ≥ 0.05), iar pragurile Verify/Block sunt alese pe zilele 78–85, separat pentru conturile
personale și cele de firmă.

Setul de date urmează regulile MIA: moneda este doar **MDL**, aliasul este numărul de telefon
(`+373` și 8 cifre), există zece scheme de plată (`p2p_alias`, `p2p_iban`, QR static și dinamic,
link de plată, Request to Pay, `b2b`, `p2g`, salarii, remitențe primite), iar limitele pentru
persoane fizice sunt **5.000 MDL pe tranzacție, 10.000 MDL pe zi și 30.000 MDL pe lună**
(`config/policy.json`, `mia_limits`). Fiecare rând are câmpul obligatoriu
`"transfer_type": "instant"`: `validate` raportează rândurile fără el sau cu altă valoare,
încărcătorul refuză fișierul, iar API-ul de simulare respinge orice alt tip, orice altă monedă decât
MDL și orice plată personală peste 5.000 MDL (422).

Scorul este un indice relativ de anomalie, **nu o probabilitate de fraudă**. Un scor mare, singur,
nu este niciodată prezentat drept fraudă confirmată.

## Pornire rapidă

Cerințe: Python ≥ 3.12 cu [uv](https://docs.astral.sh/uv/) (dezvoltat pe Python 3.14.3) și
Node.js ≥ 22.22 (dezvoltat pe Node 24.8). Nu sunt necesare servicii cloud sau credențiale.

```bash
uv sync
```

```bash
npm --prefix frontend install
```

Setul de date generat, pachetul cu modelul antrenat și raportul de evaluare sunt incluse în
repository, deci aplicația poate fi pornită imediat:

```bash
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8001
```

```bash
npm --prefix frontend run dev
```

Deschide <http://localhost:5173>. Vite redirecționează `/api` către backend-ul de pe portul 8001.
Fluxul simulat de transferuri pornește odată cu backend-ul, așa că pagina Live Transfers se
populează fără a deschide simulatorul.

Pentru a servi build-ul de producție direct din procesul backend, rulează
`npm --prefix frontend run build`, repornește uvicorn și deschide <http://127.0.0.1:8001>.

## Publicare pe Vercel (Services)

`vercel.json` definește un singur proiect Vercel cu două servicii:

| Serviciu | Rădăcină | Ce este | Cale publică |
|---|---|---|---|
| `app` | `.` | Backend FastAPI (`backend.api:app`) | `/api/*` |
| `frontend` | `frontend/` | Build static Vite, cu fallback SPA către `index.html` | restul căilor |

Browserul apelează backend-ul pe același domeniu prin URL-uri relative `/api/...`, deci niciun
serviciu nu apelează alt server și nu sunt necesare legături între servicii. Backend-ul își
păstrează prefixul `/api`, deoarece Vercel transmite calea originală a cererii. Fișierele folosite
doar offline (`storage/train.json` de 79,8 MB și `storage/validation.json`, caracteristicile din
cache, testele, documentația și sursele frontend) sunt excluse din pachetul funcției backend.
Backend-ul are nevoie doar de `storage/test.json`.

Rulează ambele servicii local cu Vercel CLI:

```bash
npx vercel dev -L
```

Pe Vercel, backend-ul rulează ca funcție pe Fluid compute. Limitarea legată de memoria procesului
(vezi mai jos) contează mai mult acolo: instanțele pot fi reciclate sau multiplicate, deci
simulările, reviziile și fluxul live sunt per instanță și de scurtă durată.

## Setul de date: trei fișiere, ca pe Kaggle

`storage/` conține **doar** trei fișiere JSON, fiecare fiind un tablou de rânduri complete (toate
câmpurile plății plus coloanele-etichetă `is_fraud`, `is_anomaly`, `fraud_type`, `anomaly_type`,
`campaign_id`):

| Fișier | Zile | Plăți evaluate | Fraude | Neobișnuite, dar legitime | Mărime | Folosire |
|---|---|---:|---:|---:|---:|---|
| `storage/train.json` | 1–70 | 70.000 | 1.500 | 3.500 | 79,8 MB | Antrenarea preprocesării, IF, LOF, DBSCAN (fără etichete) |
| `storage/validation.json` | 71–85 | 15.000 | 300 | 750 | 17,1 MB | Zilele 71–77: maparea scorurilor și ponderile; 78–85: pragurile |
| `storage/test.json` | 86–100 | 15.000 | 360 | 750 | 17,1 MB | O singură evaluare înghețată; tot el este istoricul servit de aplicație |

Pe lângă plățile evaluate, fiecare fișier conține și creditări primite (salarii, plăți de la
clienți, remitențe din diasporă, intrări în conturi „mule”) ca rânduri de context
(`assessed: false`). Zece tipuri de fraudă apar în toate partițiile (150 / 30 / 30 de plăți
fiecare). `qr_code_tampering` și `invoice_redirect` apar **doar în test** (30 fiecare), pentru a
verifica tipuri nevăzute la antrenare. Metadatele (seed, versiuni, numărători, SHA-256) sunt în
`datagen/manifest.json`, nu în `storage/`. Detalii: [docs/DATASET.md](docs/DATASET.md).

## Reconstruirea completă

```bash
uv run python -m backend.cli generate   # storage/{train,validation,test}.json + datagen/manifest.json, apoi validare
```

```bash
uv run python -m backend.cli train      # antrenare pe train.json; mapare + ponderi pe zilele 71-77, selecție pe 78-85
```

```bash
uv run python -m backend.cli evaluate   # o singură evaluare înghețată pe test.json
```

`generate` este determinist: același seed reproduce fișiere identice byte cu byte.
`uv run python -m backend.cli validate` reverifică oricând cele trei fișiere, numărătorile,
limitele MIA, partițiile, campaniile, hash-urile și amprentele, și regenerează setul într-un
director temporar pentru a confirma reproducerea byte cu byte.

## Verificări

```bash
uv run pytest -q -p no:warnings
```

```bash
uv run ruff check .
```

```bash
npm --prefix frontend test -- --run
```

```bash
npm --prefix frontend run typecheck
```

```bash
npm --prefix frontend run build
```

## Ce conține proiectul

Rezumatul tehnic al întregului proiect este în [docs/TEHNIC.md](docs/TEHNIC.md).

| Cale | Conținut |
|---|---|
| `storage/train.json`, `storage/validation.json`, `storage/test.json` | Singurele fișiere din `storage/`: setul de date sintetic MIA |
| `datagen/` | Generatorul cu seed (400 de conturi cu nume), identitățile, încărcătorul și validatorul, `manifest.json` |
| `runtime/` | Index de istoric ordonat în timp, căutare de dovezi de încredere, stare în proces, istoricul reviziilor |
| `config/limits.py` | Limitele MIA citite din `config/policy.json` |
| `ml/features.py` | Funcția unică de caracteristici, fără scurgeri de informație, folosită offline și live |
| `ml/algorithms/` | Adaptoare scikit-learn: preprocesare, IF, LOF (novelty), context DBSCAN |
| `ml/policy.py`, `config/policy.json` | Reguli, combinarea pură a scorurilor, tabelul de acțiuni bazat pe dovezi |
| `ml/training.py`, `ml/evaluation.py` | Căutare delimitată, selecția pragurilor, evaluare înghețată |
| `ml/models/` | Pachetul modelului, metadate, punct de operare, raport de antrenare |
| `artifacts/evaluation.{json,md}` | Rezultatele măsurate ale evaluării |
| `backend/` | Aplicația FastAPI, serviciul, producătorul de flux, CLI, teste |
| `frontend/` | Interfața React + TypeScript + Vite + Tailwind și testele ei |
| `docs/` | Arhitectură, algoritmi, proveniența datelor, designul interfeței |

## Rezultate măsurate (`storage/test.json`, date sintetice)

Provin din `artifacts/evaluation.md` (15.000 de plăți evaluate, 360 de fraude). Descriu doar acest
set de date generat și nu constituie dovezi de detecție sau de reducere a fraudei în lumea reală.

| Metrică | Valoare |
|---|---|
| Fraude trimise la Verify sau Block | 76,7% (276 din 360; IÎ 95% 70,8–82,8%) |
| Fraude blocate | 30,8% (111 din 360; 24 prin dovezi confirmate, 87 preventiv) |
| Precizia blocărilor | 96,5% din 115 blocări (IÎ 95% 91,4–98,6%; obiectiv 99%: **neatins**) |
| Rata de blocări false pentru plăți legitime | 0,03% (4 din 14.640; obiectiv ≤ 0,1%: atins) |
| Rata Verify | 4,58% (obiectiv ≤ 5%: atins; 4,55% pe validare) |
| Doar model + reguli, Verify sau Block | 75,8% |
| Tipurile doar-în-test (`qr_code_tampering`, `invoice_redirect`) | 66,7% semnalate (40 din 60) |
| Conturi holdout (nefolosite la antrenare) | 80,0% recall pe fraude, 2,2% din plățile legitime revizuite |
| Latența unei plăți | p50 4,5 ms, p95 4,7 ms (în proces) |

### Recall pe tip de fraudă

| Tip de fraudă (30 de plăți fiecare) | Verify sau Block | Block |
|---|---:|---:|
| `account_takeover_phishing` | 76,7% | 70,0% |
| `sim_swap_takeover` | 100,0% | 86,7% |
| `vishing_impersonation` | 70,0% | 13,3% |
| `request_to_pay_scam` | 100,0% | 50,0% |
| `marketplace_scam` | 70,0% | 16,7% |
| `investment_scam` | 36,7% | 0,0% |
| `romance_scam` | 36,7% | 3,3% |
| `money_mule` | 96,7% | 33,3% |
| `limit_structuring` | 100,0% | 33,3% |
| `coordinated_beneficiary` | 100,0% | 60,0% |
| `qr_code_tampering` (doar în test) | 83,3% | 3,3% |
| `invoice_redirect` (doar în test) | 50,0% | 0,0% |

Escrocheriile lente (investiții, romantice) trimit sume moderate unei singure persoane noi, de pe
dispozitivul propriu al victimei, și seamănă cu o relație obișnuită; de aceea sunt prinse rar.
Plățile legitime aproape de limita de 5.000 MDL ajung aproape mereu la verificare (96,7% dintre
achizițiile planificate aproape de limită), dar sunt blocate rar, pentru că o blocare cere două
familii de semnale independente.

### Ce valoare are procentul de risc

Scorul ordonează cât de neobișnuită este o plată; randamentul lui este măsurat pe intervale de scor
în perioada de test etichetată (`score_bands` în `artifacts/evaluation.json`, afișat la fiecare
investigație și în Algorithm Results):

| Interval de scor | Plăți | Fraude | Rata de fraudă observată | Zonă |
|---|---:|---:|---:|---|
| 0–50 | 10.255 | 6 | 0,06% | Permit |
| 50–69 | 3.837 | 70 | 1,8% | Permit |
| 69–80 | 636 | 93 | 14,6% | Verify (firme de la 69, persoane de la 71) |
| 80–91 | 170 | 93 | 54,7% | Verify |
| 91–100 | 102 | 98 | 96,1% | Zona Block (cu două familii de semnale) |

Rata de bază a fraudei este 2,4%, deci o plată cu scor 80–91 are o probabilitate de aproximativ 23×
mai mare de a fi fraudă decât una medie. Semnalarea tuturor plăților de la 69 în sus prinde 78,9%
din fraude, atingând 6,05% din trafic.

Vezi [docs/ALGORITHMS.md](docs/ALGORITHMS.md) pentru detalierea pe componente și pe tipuri și
pentru compromisurile din spatele acestor cifre.

## Limitări de știut înainte de utilizare

- **Memoria procesului.** Simulările și fluxul live trăiesc într-un singur proces backend. O
  repornire le șterge; fișierele de bază și modelele rămân.
- **Istoricul reviziilor este persistent.** Fiecare aprobare sau respingere este scrisă în
  `runtime/state/review_history.sqlite3` (se poate schimba cu `SECFLOW_STATE_DIR`; pe Vercel ajunge
  în `/tmp`, care este per instanță). După o repornire, reviziile cererilor de bază sunt reaplicate,
  iar pagina Review History listează în continuare toate deciziile, inclusiv reviziile cererilor
  simulate. Mai mulți workeri sau mai multe instanțe ar avea fiecare o stare separată, nepartajată.
  Rulează un singur proces (`uvicorn` fără `--workers`).
- **Totul este sintetic.** Conturile, contrapărțile, aliasurile de telefon, plățile și dovezile de
  securitate sunt generate. Numele sunt plauzibile, dar nu aparțin niciunei persoane sau firme
  verificate, iar numerele de telefon sunt aleatorii.
- **Aplicația vede doar istoricul de test.** Serviciul încarcă numai `storage/test.json`, deci în
  primele zile de test plățile nu au istoric anterior în aplicație, spre deosebire de evaluare, care
  vede zilele 1–85 ca istoric. Scorurile afișate în aplicație pentru acele plăți pot diferi de cele
  măsurate în evaluare.
- **Blocările preventive sunt rare intenționat.** Punctul de operare validat păstrează cazurile
  ambigue cu risc mare în Verify în loc să le blocheze; vezi compromisul descris în documentație.
- **Fără autentificare.** Numele revizorului este doar o valoare afișată pentru acest flux local.
