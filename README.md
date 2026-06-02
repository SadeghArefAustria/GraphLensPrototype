# GraphLens Prototype

Extract knowledge graphs from PDF documents with Claude and load them into Neo4j.

```
PDF  →  Claude (Files API + structured output)  →  KG JSON  →  Neo4j
```

---

## Project layout

```
GraphLensPrototype/
├── graphlens/               # importable library
│   ├── __init__.py
│   ├── extractor.py         # Claude-based KG extraction
│   └── neo4j_loader.py      # Neo4j integration
├── scripts/                 # CLI entry points
│   ├── extract_kg.py        # extract from PDF → JSON
│   └── load_to_neo4j.py     # JSON → Neo4j
├── data/
│   ├── input/               # put your PDFs here  (gitignored)
│   └── output/              # extracted JSONs land here  (gitignored)
├── examples/
│   └── results.json         # sample extraction output
├── .env.example             # environment variable template
├── requirements.txt
└── README.md
```

---

## Quick start

### 1 — Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Set environment variables

```bash
# Copy the template and fill in your values
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Minimum required:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Get one at console.anthropic.com |
| `NEO4J_URI` | e.g. `bolt://localhost:7687` |
| `NEO4J_USER` | e.g. `neo4j` |
| `NEO4J_PASSWORD` | Your Neo4j password |

On Windows PowerShell you can also set them inline:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 3 — Extract a knowledge graph

```bash
python scripts/extract_kg.py data/input/my_paper.pdf --out data/output/results.json
```

Options:

| Flag | Description |
|---|---|
| `--out PATH` | Write JSON to file (default: stdout) |
| `--file-id ID` | Reuse a previously uploaded file, skip upload |
| `--keep-file` | Don't delete the file from the Files API after extraction |

### 4 — Load into Neo4j

```bash
python scripts/load_to_neo4j.py data/output/results.json \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password your-password
```

Then open **Neo4j Browser** at `http://localhost:7474` and explore:

```cypher
-- Full graph
MATCH (n:Entity)-[r]->(m:Entity) RETURN n, r, m

-- All persons
MATCH (n:PERSON) RETURN n

-- Relationships for a specific entity
MATCH (n:Entity {name: "TU Graz"})-[r]->(m) RETURN n, r, m
```

---

## Using the library directly

```python
import anthropic
from graphlens import upload_pdf, extract, KGLoader

client  = anthropic.Anthropic()

# 1. Upload and extract
file_id = upload_pdf(client, Path("paper.pdf"))
result  = extract(client, file_id)   # {"entities": [...], "relations": [...]}
client.beta.files.delete(file_id)

# 2. Load into Neo4j
with KGLoader("bolt://localhost:7687", "neo4j", "password") as loader:
    nodes, rels = loader.load(result)
    print(f"Loaded {nodes} nodes and {rels} relations.")
```

---

## KG JSON format

```json
{
  "entities": [
    {"name": "TU Graz", "type": "ORG", "description": "..."}
  ],
  "relations": [
    {"subject": "TU Graz", "predicate": "PARTNERED_WITH",
     "object": "Magna",   "evidence": "..."}
  ]
}
```

Entity types: `PERSON` · `ORG` · `LOCATION` · `EVENT` · `CONCEPT` · `PRODUCT` · `OTHER`

See [examples/results.json](examples/results.json) for a full sample.

---

## Ideas for further improvements

- [ ] Batch-process multiple PDFs in one run
- [ ] Support plain-text and web-page inputs alongside PDFs
- [ ] Add a confidence score to each extracted triple
- [ ] Persist `file_id` to avoid re-uploading the same document
- [ ] Query the graph with natural language via Claude + Cypher generation
- [ ] Export to other graph formats (RDF/Turtle, GraphML)
- [ ] Add a FastAPI service layer so the pipeline can be called over HTTP