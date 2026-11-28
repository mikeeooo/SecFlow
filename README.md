# SecFlow

SecFlow este un proiect universitar de detecție a fraudei. Analizează plăți instant **MIA Plăți
Instant** (sistemul de plăți instant al Băncii Naționale a Moldovei, în lei moldovenești, MDL) și
explică ce decide. Fiecare plată primește un indice de risc
de anomalie construit din trei algoritmi scikit-learn (IsolationForest, LocalOutlierFactor, DBSCAN)
și din reguli explicite de comportament. O politică de tip „mai întâi dovezile” transformă apoi
acest indice în **Permit** (permite), **Verify** (verifică) sau **Block** (blochează), iar fiecare
decizie își arată calculul, codurile de motiv și sursa acțiunii.

```text
risk_score = 100 × (0.65 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.30 × rules)
```

Cei trei algoritmi sunt nesupervizați și sunt antrenați doar pe `train.json`, fără etichete.
Ponderile sunt fixe, în `config/policy.json`. Etichetele din `validation.json` servesc doar la
alegerea pragurilor Verify și Block, iar etichetele din `test.json` doar la evaluarea finală.

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
doar offline (`storage/train.json` de 79,8 MB, testele, documentația și sursele frontend) sunt
excluse din pachetul funcției backend. Backend-ul folosește `storage/validation.json` ca istoric și
`storage/test.json` ca plăți afișate.

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
| `storage/validation.json` | 71–85 | 15.000 | 300 | 750 | 17,1 MB | Referința scorurilor IF și LOF (fără etichete), apoi alegerea pragurilor |
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
uv run python -m backend.cli train      # antrenare pe train.json, alegerea pragurilor pe validation.json
```

```bash
uv run python -m backend.cli evaluate   # o singură evaluare înghețată pe test.json
```

`generate` este determinist: același seed reproduce fișiere identice byte cu byte, iar un test
verifică asta. `uv run python -m backend.cli validate` reverifică oricând cele trei fișiere,
hash-urile, numărul de plăți, moneda și tipul transferului, limitele MIA, tipurile de fraudă pe
partiții, eliminarea etichetelor și rândurile copiate între fișiere. Antrenarea durează în jur de
5 secunde, iar evaluarea în jur de 7 secunde.

## Verificări

```bash
uv run pytest -q -p no:warnings
```

```bash
uv run ruff check .
```

```bash
uv run python scripts/api_contract.py --check
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
| `config/limits.py` | Limitele MIA citite din `config/policy.json` |
| `ml/history.py`, `ml/features.py` | Istoricul ordonat în timp și funcția unică de caracteristici, folosită offline și live |
| `ml/model.py` | Preprocesarea, maparea scorurilor, contextul DBSCAN și pachetul modelului |
| `ml/policy.py`, `config/policy.json` | Ponderile fixe, regulile, scorul de risc și tabelul de decizie |
| `ml/evidence.py`, `ml/assess.py` | Dovezile de securitate și calea unică de la plată la decizie |
| `ml/training.py`, `ml/evaluation.py` | Antrenarea cu alegerea pragurilor și evaluarea înghețată |
| `scripts/api_contract.py` | Verificarea formei răspunsurilor API de care depinde interfața |
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
| Fraude trimise la Verify sau Block | 78,6% (283 din 360; IÎ Wilson 95% 74,1–82,5%) |
| Fraude blocate | 34,7% (125 din 360; 24 prin dovezi confirmate, 101 preventiv) |
| Precizia blocărilor | 96,9% din 129 blocări (IÎ 95% 92,3–98,8%; obiectiv 99%: **neatins**) |
| Rata de blocări false pentru plăți legitime | 0,03% (4 din 14.640; obiectiv ≤ 0,1%: atins) |
| Rata Verify | 4,49% (obiectiv ≤ 5%: atins; 3,39% pe validare) |
| Doar model + reguli, Verify sau Block | 78,1% |
| Tipurile doar-în-test (`qr_code_tampering`, `invoice_redirect`) | 56,7% semnalate (34 din 60) |
| Conturi noi (40 de conturi, 678 de plăți) | toate cele 30 de fraude semnalate, 23,1% din plățile legitime revizuite |
| Latența unei plăți | p50 4,6 ms, p95 5,3 ms (în proces) |

### Recall pe tip de fraudă

| Tip de fraudă (30 de plăți fiecare) | Verify sau Block | Block |
|---|---:|---:|
| `account_takeover_phishing` | 93,3% | 70,0% |
| `sim_swap_takeover` | 100,0% | 86,7% |
| `vishing_impersonation` | 76,7% | 20,0% |
| `request_to_pay_scam` | 100,0% | 56,7% |
| `marketplace_scam` | 73,3% | 20,0% |
| `investment_scam` | 50,0% | 0,0% |
| `romance_scam` | 40,0% | 3,3% |
| `money_mule` | 96,7% | 26,7% |
| `limit_structuring` | 100,0% | 46,7% |
| `coordinated_beneficiary` | 100,0% | 73,3% |
| `qr_code_tampering` (doar în test) | 80,0% | 13,3% |
| `invoice_redirect` (doar în test) | 33,3% | 0,0% |

Escrocheriile lente (investiții, romantice) trimit sume moderate unei singure persoane noi, de pe
dispozitivul propriu al victimei, și seamănă cu o relație obișnuită. De aceea sunt prinse rar.
Plățile legitime aproape de limita de 5.000 MDL ajung aproape mereu la verificare (30 din 30 de
achiziții planificate aproape de limită), dar sunt blocate rar, pentru că o blocare cere două
familii de semnale independente.

### Ce valoare are procentul de risc

Scorul ordonează cât de neobișnuită este o plată. Valoarea lui este măsurată pe intervale de scor în
perioada de test etichetată (`score_bands` în `artifacts/evaluation.json`, afișat la fiecare
investigație și în Algorithm Results):

| Interval de scor | Plăți | Fraude | Rata de fraudă observată | Zonă |
|---|---:|---:|---:|---|
| 0–50 | 10.446 | 6 | 0,06% | Permit |
| 50–68 | 3.769 | 73 | 1,9% | Permit |
| 68–80 | 562 | 108 | 19,2% | Verify |
| 80–86 | 94 | 53 | 56,4% | Verify |
| 86–100 | 129 | 120 | 93,0% | Zona Block (cu două familii de semnale) |

Rata de bază a fraudei este 2,4%. O plată cu scor 80–86 este fraudă de aproximativ 23 de ori mai des
decât una medie. Semnalarea tuturor plăților de la 68 în sus prinde 78,1% din fraude și atinge 5,2%
din trafic.

Vezi [docs/ALGORITHMS.md](docs/ALGORITHMS.md) pentru detalierea pe componente și pe tipuri și
pentru compromisurile din spatele acestor cifre.

## Limitări de știut înainte de utilizare

- **Memoria procesului.** Simulările, fluxul live și reviziile trăiesc într-un singur proces
  backend. O repornire le șterge; fișierele de bază și modelele rămân. Mai mulți workeri sau mai
  multe instanțe Vercel ar avea fiecare o stare separată. Rulează un singur proces (`uvicorn` fără
  `--workers`).
- **Totul este sintetic.** Conturile, contrapărțile, aliasurile de telefon, plățile și dovezile de
  securitate sunt generate. Numele sunt plauzibile, dar nu aparțin niciunei persoane sau firme
  verificate, iar numerele de telefon sunt aleatorii.
- **Aplicația are mai puțin istoric decât evaluarea.** Serviciul încarcă `storage/validation.json`
  doar ca istoric și afișează plățile din `storage/test.json`. Vede deci zilele 71–100, iar evaluarea
  vede zilele 1–100. Caracteristicile privesc cel mult 30 de zile în urmă, așa că rezultatele sunt
  apropiate: circa 5% Verify în aplicație și 4,49% în raport.
- **Blocările preventive sunt rare intenționat.** Punctul de operare validat păstrează cazurile
  ambigue cu risc mare în Verify în loc să le blocheze; vezi compromisul descris în documentație.
- **Testul nu este complet neatins.** O primă rulare a trimis 5,81% din plățile de test la Verify.
  Rezerva pentru conturile noi și ponderile a cinci reguli au fost alese apoi pe cifre din validare,
  dar motivul căutării a venit din rezultatul pe test.
- **Fără autentificare.** Numele revizorului este doar o valoare afișată pentru acest flux local.
