# SecFlow

SecFlow analizează cereri de plată de tip instant și explică ce decide. Fiecare cerere primește un
indice de risc de anomalie construit din trei algoritmi scikit-learn (IsolationForest,
LocalOutlierFactor, DBSCAN) și din reguli explicite de comportament. O politică de tip „mai întâi
dovezile” transformă apoi acest indice în **Permit** (permite), **Verify** (verifică) sau **Block**
(blochează), iar fiecare decizie își arată calculul, codurile de motiv și sursa acțiunii.

```text
risk_score = 100 × (0.80 × IF + 0.05 × LOF + 0.00 × DBSCAN + 0.15 × rules)
```

Ponderile sunt învățate pe perioada de calibrare (căutare exhaustivă pe simplex, cu LOF menținut la
≥ 0.05), iar pragurile Verify/Block sunt alese separat pentru conturile personale și cele de firmă.

SecFlow evaluează doar transferuri instant. Fiecare cerere din `requests.jsonl` are câmpul
obligatoriu `"transfer_type": "instant"`: `validate` raportează rândurile fără el sau cu altă
valoare, `load_dataset` refuză fișierul, iar API-ul de simulare respinge orice alt tip (422).

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

Seturile de date generate, pachetul cu modelul antrenat și raportul de evaluare sunt incluse în
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
doar offline (corpusul de antrenare de 88 MB, caracteristicile din cache, testele, documentația și
sursele frontend) sunt excluse din pachetul funcției backend.

Rulează ambele servicii local cu Vercel CLI:

```bash
npx vercel dev -L
```

Pe Vercel, backend-ul rulează ca funcție pe Fluid compute. Limitarea legată de memoria procesului
(vezi mai jos) contează mai mult acolo: instanțele pot fi reciclate sau multiplicate, deci
simulările, reviziile și fluxul live sunt per instanță și de scurtă durată.

## Reconstruirea completă

```bash
uv run python -m backend.cli generate   # ambele seturi de date cu seed, manifeste, apoi validare
```

```bash
uv run python -m backend.cli train      # antrenare pe zilele 1-60, mapare + învățarea ponderilor pe 61-70, selecție pe 71-80
```

```bash
uv run python -m backend.cli evaluate   # o singură evaluare înghețată pe zilele 81-100
```

`generate` este determinist: aceleași seed-uri reproduc fișiere identice byte cu byte (verificat
prin SHA-256 în `manifest.json`). `uv run python -m backend.cli validate` reverifică oricând
numărătorile, partițiile, integritatea campaniilor, hash-urile și amprentele dintre seturile de date.

## Verificări

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

## Ce conține proiectul

| Cale | Conținut |
|---|---|
| `storage/generator.py`, `storage/identities.py` | Generator cu seed pentru ambele seturi de date (400 de conturi cu nume) |
| `storage/data/training/` | Corpus de 100.000 de cereri (seed 42), etichete, partiții, dovezi, manifest |
| `storage/data/demo/` | Istoric demonstrativ independent de 10.000 de cereri (seed 43) |
| `storage/history.py`, `storage/evidence.py`, `storage/memory.py` | Index de istoric ordonat în timp, căutare de dovezi de încredere, stare în proces |
| `ml/features.py` | Funcția unică de caracteristici, fără scurgeri de informație, folosită offline și live |
| `ml/algorithms/` | Adaptoare scikit-learn: preprocesare, IF, LOF (novelty), context DBSCAN |
| `ml/policy.py`, `config/policy.json` | Reguli, combinarea pură a scorurilor, tabelul de acțiuni bazat pe dovezi |
| `ml/training.py`, `ml/evaluation.py` | Căutare delimitată, selecția pragurilor, evaluare înghețată |
| `ml/models/` | Pachetul modelului, metadate, punct de operare, raport de antrenare |
| `artifacts/evaluation.{json,md}` | Rezultatele măsurate ale evaluării |
| `backend/` | Aplicația FastAPI, serviciul, producătorul de flux, CLI, teste |
| `frontend/` | Interfața React + TypeScript + Vite + Tailwind și testele ei |
| `docs/` | Arhitectură, algoritmi, proveniența datelor, designul interfeței |

## Rezultate măsurate (perioada finală de test, date sintetice)

Provin din `artifacts/evaluation.md`. Descriu doar acest set de date generat și nu constituie dovezi
de detecție sau de reducere a fraudei în lumea reală.

| Metrică | Valoare |
|---|---|
| Fraude trimise la Verify sau Block | 76,2% (183 din 240; IÎ 95% 68,2–83,2%) |
| Fraude blocate | 29,6% (71 din 240; 34 prin dovezi confirmate, 37 preventiv) |
| Precizia blocărilor | 97,3% din 73 de blocări (IÎ 95% 90,6–99,3%; obiectiv 99%: **neatins**) |
| Rata de blocări false pentru cereri legitime | 0,01% (2 din 19.760) |
| Rata Verify | 5,89% (obiectiv de validare ≤ 5%: **neatins** pe test; 3,68% pe validare) |
| Doar model + reguli, Verify sau Block | 75,0% |
| Scenariu nevăzut `invoice_redirect` | 12,5% semnalate (5 din 40) |
| Conturi holdout (nefolosite la antrenare) | 70,4% recall pe fraude, 2,7% din cererile legitime revizuite |
| Latența unei cereri | p50 5,5 ms, p95 6,2 ms (în proces) |

### Ce valoare are procentul de risc

Scorul ordonează cât de neobișnuită este o cerere; randamentul lui este măsurat pe intervale de scor
în perioada de test etichetată (`score_bands` în `artifacts/evaluation.json`, afișat la fiecare
investigație și în Algorithm Results):

| Interval de scor | Cereri | Fraude | Rata de fraudă observată | Zonă |
|---|---:|---:|---:|---|
| 0–50 | 10.955 | 0 | 0% | Permit |
| 50–70 | 4.645 | 2 | 0,04% | Permit |
| 70–80 | 2.448 | 22 | 0,9% | Permit |
| 80–83 | 809 | 15 | 1,9% | Verify (personal) |
| 83–90 | 946 | 74 | 7,8% | Verify |
| 90–99 | 173 | 104 | 60,1% | Verify / Block |
| 99–100 | 24 | 23 | 95,8% | Zona Block |

Rata de bază a fraudei este 1,2%, deci o cerere cu scor 90–99 are o probabilitate de aproximativ 50×
mai mare de a fi fraudă decât una medie. Semnalarea tuturor cererilor de la 83 în sus prinde 83,8%
din fraude, atingând 5,71% din trafic.

Vezi [docs/ALGORITHMS.md](docs/ALGORITHMS.md) pentru detalierea pe componente și pe scenarii și
pentru compromisurile din spatele acestor cifre.

## Limitări de știut înainte de utilizare

- **Memoria procesului.** Simulările și fluxul live trăiesc într-un singur proces backend. O
  repornire le șterge; fișierele de bază și modelele rămân.
- **Istoricul reviziilor este persistent.** Fiecare aprobare sau respingere este scrisă în
  `storage/state/review_history.sqlite3` (se poate schimba cu `SECFLOW_STATE_DIR`; pe Vercel ajunge
  în `/tmp`, care este per instanță). După o repornire, reviziile cererilor de bază sunt reaplicate,
  iar pagina Review History listează în continuare toate deciziile, inclusiv reviziile cererilor
  simulate. Mai mulți workeri sau mai multe instanțe ar avea fiecare o stare separată, nepartajată.
  Rulează un singur proces (`uvicorn` fără `--workers`).
- **Totul este sintetic.** Conturile, contrapărțile, plățile și dovezile de securitate sunt
  generate. Numele sunt plauzibile, dar nu aparțin niciunei persoane sau firme verificate.
- **Istoricul demo este mai rar** (100 de cereri pe zi față de 1.000 în corpusul de antrenare), deci
  ponderea Verify (≈19%) este mai mare decât rata din perioada de test: IsolationForest, care are cea
  mai mare parte din pondere, interpretează istoricul rar ca neobișnuit. Este istoric de produs, nu
  un benchmark.
- **Blocările preventive sunt rare intenționat.** Punctul de operare validat păstrează cazurile
  ambigue cu risc mare în Verify în loc să le blocheze; vezi compromisul descris în documentație.
- **Fără autentificare.** Numele revizorului este doar o valoare afișată pentru acest flux local.
