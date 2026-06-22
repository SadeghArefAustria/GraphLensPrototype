# KGIntel Corpus Collection Guide — WP1
**Target:** Minimum 200 documents covering the driving simulator market (DACH focus, English language)
**Document types:** Academic PDFs, procurement notices, news articles, job postings

---

## Source 1 — Academic Publications (target: 80–100 documents)

### Google Scholar search queries
Copy each query exactly into scholar.google.com. Download PDF where available.
Set date range: 2018–2026.

```
"driving simulator" AND ("customer" OR "market" OR "application")
"driving simulator" AND ("autonomous driving" OR "ADAS") site:tu-graz.ac.at
"driving simulation" AND "research" AND "Austria" OR "Germany"
"driving simulator" AND ("procurement" OR "acquisition" OR "training")
"high-fidelity driving simulator" AND "automotive"
"vehicle-in-the-loop" OR "driver-in-the-loop" simulation
"driving simulation center" AND ("university" OR "institute")
"simulator sickness" OR "motion platform" AND "research"
"ADAS validation" AND "driving simulator"
"autonomous vehicle testing" AND "simulation" AND ("Europe" OR "DACH")
```

### Semantic Scholar API (programmatic, free)
```
https://api.semanticscholar.org/graph/v1/paper/search?query=driving+simulator+automotive+research&fields=title,authors,year,venue,abstract,url&limit=100
```

### Key journals to manually browse:
- Transportation Research Part F (Elsevier)
- IEEE Transactions on Intelligent Vehicles
- Accident Analysis and Prevention
- Simulation Modelling Practice and Theory
- Vehicle System Dynamics

---

## Source 2 — Procurement Notices (target: 30–50 documents)

### TED (Tenders Electronic Daily — EU official procurement portal)
URL: https://ted.europa.eu/en/search/result
Search terms to use:
```
"driving simulator"
"Fahrsimulator"  (German — still useful for entity extraction)
"simulation equipment" AND "automotive"
"driver training simulator"
"vehicle dynamics simulator"
```
CPV codes to filter by:
- 34153000-9 (Test track equipment)
- 38412000-6 (Thermometers / simulators)
- 72212000-4 (Programming services — often covers simulation software)

Download the HTML or PDF version of each notice.

### Austrian procurement portal
URL: https://www.auftrag.at OR https://www.beschaffung.gv.at
Search: "Fahrsimulator" OR "Driving Simulator"

### German procurement portal
URL: https://www.bund.de/IMPORTE/Ausschreibungen/
Search: "Fahrsimulator" OR "Driving Simulator"

---

## Source 3 — News Articles and Press Releases (target: 40–60 documents)

### EurekAlert (academic press releases — excellent for org/tech entities)
URL: https://www.eurekalert.org
Search: "driving simulator" — filter to 2020–2026
Save as: copy full text into a .txt file named by date and organisation

### Google News
Search queries:
```
"driving simulator" site:tugraz.at OR site:avl.com OR site:magna.com
"driving simulation center" 2024 OR 2025 OR 2026
"Fahrsimulator" Austria OR Germany OR Switzerland
"autonomous vehicle simulator" research institute
```

### Direct institutional press release pages to monitor:
- https://www.tugraz.at/en/tu-graz/services/news-stories/media-service/
- https://www.avl.com/newsroom
- https://www.magna.com/en/about-magna/news
- https://www.ait.ac.at/en/news/
- https://www.dlr.de/en/latest/news (German Aerospace Center)
- https://www.ifsttar.fr (French transport research)

---

## Source 4 — Job Postings (target: 20–30 documents)

These are critical demand signals — a company hiring simulation engineers is a potential customer.

### LinkedIn (manual search)
Search: "driving simulator engineer" OR "vehicle simulation" OR "ADAS simulation"
Filter: DACH region (Austria, Germany, Switzerland)
Filter: Posted last 90 days
Save as PDF (Ctrl+P → Save as PDF) or copy text into .txt file

### Indeed DACH
URL: https://de.indeed.com
Search: "Fahrsimulator" OR "Driving Simulator" OR "ADAS Simulation"
Location: Deutschland, Österreich, Schweiz

### Company career pages to monitor:
- BMW: https://www.bmwgroup.jobs/en
- Bosch: https://www.bosch.com/careers/
- Continental: https://www.continental.com/en/career/
- ZF: https://jobs.zf.com
- Volkswagen: https://www.volkswagenag.com/en/careers.html
- Porsche Engineering: https://www.porsche-engineering.com/en/career/

---

## File Naming Convention

Use this naming convention for all saved documents:

```
[TYPE]_[SOURCE]_[DATE]_[SHORTNAME].[ext]

Examples:
PDF_ScholarTUGraz_20240315_DriverDrowsiness.pdf
PROC_TED_20250101_SimulatorTrainingAT.html
NEWS_EurekAlert_20260127_MagnaTUGrazSimCenter.txt
JOB_LinkedIn_20260510_BMWSimEngineer.txt
```

Types: PDF, PROC (procurement), NEWS, JOB

---

## Tracking Spreadsheet Columns

Create a spreadsheet with these columns for every collected document:

| Column | Description |
|---|---|
| doc_id | Sequential ID (DOC_001, DOC_002, ...) |
| filename | Saved filename |
| type | PDF / PROC / NEWS / JOB |
| source | Google Scholar / TED / EurekAlert / LinkedIn / etc. |
| date_published | Publication date (YYYY-MM-DD) |
| title | Full document title |
| language | EN / DE |
| organisations_mentioned | Comma-separated list (fill after reading) |
| annotated | Yes / No |
| notes | Any notes |

---

## Priority Documents (collect first)

These specific documents are high-priority — they are known to contain rich entity-relation content relevant to the driving simulator domain:

1. TU Graz / Magna Advanced Driving Simulation Center press release (EurekAlert, Jan 2026) ✓ already in proposal figures
2. IEEE ITSC proceedings 2023–2025 — search for TU Graz author papers
3. AVL List GmbH corporate publications on ADAS simulation
4. Driving Simulation Association (DSA) member list and publications: https://www.drivingsimulationassociation.org
5. National Road Safety Authority (NRSA) procurement notices for driver training simulators
