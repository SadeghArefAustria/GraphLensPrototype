# GraphLens Prototype

GraphLens extracts entities and typed relationships from PDFs or scraped web
pages with Claude, stores the resulting knowledge graph in Neo4j, and provides
utilities for graph analysis and link prediction.

```text
PDF or web page -> Claude extraction -> KG JSON -> Neo4j
                                          |
                                          +-> NetworkX / PyKEEN / PyG
```

## Features

- Extract a structured knowledge graph from a PDF through the Anthropic Files
  API, with optional verification and page-range chunking.
- Scrape web pages into clean text, respecting `robots.txt` by default, then
  extract a graph with the same pipeline.
- Create stable document metadata and provenance for relations, including the
  source document ID, source link, evidence, page, and character span when
  available.
- Load graphs idempotently into Neo4j.
- Use the Streamlit interface to upload a PDF, extract and load its graph, and
  inspect an interactive document-specific visualization.
- Build `NetworkX`, PyKEEN, and PyTorch Geometric representations for graph
  analysis and link prediction.

## Project layout

```text
GraphLensPrototype/
├── app.py                         # Streamlit upload, extraction, and graph viewer
├── graphlens/                     # Python package
│   ├── extractor.py               # PDF and text KG extraction with Claude
│   ├── scraper.py                 # Polite web-page scraping and text cleanup
│   ├── metadata.py                # PDF metadata and stable document IDs
│   ├── neo4j_loader.py            # Idempotent Neo4j loader
│   └── ml/
│       ├── graph_builder.py       # KGGraph and graph-format conversions
│       └── link_prediction.py     # Heuristic and PyKEEN predictors
├── scripts/
│   ├── extract_kg.py              # PDF-to-KG JSON command-line interface
│   └── load_to_neo4j.py           # KG JSON-to-Neo4j command-line interface
├── data/
│   ├── input/                     # Local source PDFs (ignored by Git)
│   └── output/                    # Generated KG JSON files (ignored by Git)
├── examples/results.json          # Example extracted graph
├── notebooks/link_prediction.ipynb
├── docs/WP2_corpus_collection_guide.md
├── requirements.txt
└── pyproject.toml
```

## Requirements

- Python 3.10 or later
- An Anthropic API key for extraction
- A running Neo4j instance for loading or using the web interface

## Installation

Create and activate a virtual environment, then install the package.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Install an optional extra when needed:

```powershell
pip install -e ".[ui]"    # Streamlit interface and PyVis visualization
pip install -e ".[ml]"    # PyKEEN and PyTorch embedding models
pip install -e ".[pyg]"   # PyTorch Geometric support
pip install -e ".[all]"   # All optional extras
```

`requirements.txt` is also available for installing the core dependencies plus
the UI dependencies directly.

## Configuration

Copy the sample configuration and enter your credentials:

```powershell
Copy-Item .env.example .env
```

| Variable | Used by | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | extraction and Streamlit UI | Anthropic API key |
| `NEO4J_URI` | Streamlit UI | Bolt URI, such as `bolt://localhost:7687` |
| `NEO4J_USER` | Streamlit UI | Neo4j username |
| `NEO4J_PASSWORD` | Streamlit UI | Neo4j password |

The Streamlit app loads `.env` automatically. For the command-line extraction
script, make `ANTHROPIC_API_KEY` available in the shell environment. The
Neo4j loader accepts its connection settings as command-line options.

```powershell
$env:ANTHROPIC_API_KEY = "your_anthropic_api_key"
```

Never commit `.env` or source documents containing sensitive information.

## PDF workflow

Place a PDF under `data/input/`, extract its graph, then load the saved JSON
into Neo4j.

```powershell
python scripts/extract_kg.py data/input/paper.pdf --out data/output/paper.json

python scripts/load_to_neo4j.py data/output/paper.json `
  --uri bolt://localhost:7687 --user neo4j --password your-password
```

Useful extraction options:

```powershell
# Add a domain hint and run a second extraction pass to improve recall.
python scripts/extract_kg.py data/input/paper.pdf `
  --domain "academic research" --verify --out data/output/paper.json

# Preserve a source URL on every relationship.
python scripts/extract_kg.py data/input/paper.pdf `
  --source-link https://example.org/paper --out data/output/paper.json

# Read a document ID and source link from an existing metadata JSON file.
python scripts/extract_kg.py data/input/paper.pdf `
  --metadata data/input/paper-metadata.json --out data/output/paper.json

# Process a large PDF in page chunks and retain the intermediate artifacts.
python scripts/extract_kg.py data/input/paper.pdf --chunk-pages 5 --save-chunks `
  --out data/output/paper.json
```

Use `--file-id` to reuse a previously uploaded Anthropic Files API document,
`--keep-file` to retain an upload after extraction, and `--save-text` to write
a human-readable extraction summary alongside `--out`.

## Web interface

After configuring `.env` and installing the UI extra, start the Streamlit app:

```powershell
streamlit run app.py
```

The interface accepts PDF uploads up to 100 MB, optionally takes a domain
hint and verification pass, writes each result to
`data/output/<document-id>.json`, loads it into Neo4j, and displays an
interactive graph limited to relationships from the uploaded document.

## Extracting from web pages

The scraper returns cleaned article text suitable for `extract_from_text`.
It checks `robots.txt` by default and caps extracted content at 120,000
characters per page.

```python
import anthropic

from graphlens import extract_from_text
from graphlens.scraper import scrape

page = scrape("https://example.org/article")
if page.ok:
    client = anthropic.Anthropic()
    result = extract_from_text(client, page.text, title=page.title, domain="news")
```

Use `scrape_many([...], delay=1.0)` to fetch pages sequentially with a polite
delay.

## Machine-learning utilities

`KGGraph` converts extraction JSON into an integer-encoded graph, supports
merging and train/validation/test splits, and exports formats for NetworkX,
PyKEEN, and PyTorch Geometric.

```python
from graphlens.ml import KGGraph, HeuristicPredictor

kg = KGGraph.from_json("data/output/paper.json")
train, valid, test = kg.train_test_split(test_size=0.2, valid_size=0.1)

predictor = HeuristicPredictor(kg, method="common_neighbors")
metrics = predictor.evaluate(test, train_triples=train)
predictions = predictor.predict_tails("TU Graz", top_k=5)
```

Available heuristic methods are `common_neighbors`, `jaccard`, and
`adamic_adar`. With the `ml` extra installed, `PyKEENPredictor` supports
embedding models such as `TransE`, `RotatE`, `DistMult`, `ComplEx`, and
`ConvE`. With the `pyg` extra, use `kg.to_pyg_data()` to create a PyTorch
Geometric `Data` object.

The notebook at `notebooks/link_prediction.ipynb` is an interactive starting
point for link-prediction experiments.

## Neo4j queries

```cypher
// Entire graph
MATCH (source:Entity)-[relationship]->(target:Entity)
RETURN source, relationship, target

// Entities of one type
MATCH (person:PERSON)
RETURN person

// Relationships from one source document
MATCH (source:Entity)-[relationship]->(target:Entity)
WHERE relationship.source_file_id = "your-document-id"
RETURN source, relationship, target
```

The loader creates a unique constraint on `:Entity(name)`, merges entities by
name, and preserves validated entity-type labels and relationship predicates.

## KG JSON format

Extraction results contain `entities` and `relations`. PDF results also
include a top-level `doc_id`; relations can include provenance fields such as
`source_sentence`, `page`, `char_span`, `source_file_id`, and
`source_file_link`.

```json
{
  "entities": [
    {"name": "TU Graz", "type": "ORG", "description": "A university."}
  ],
  "relations": [
    {
      "subject": "TU Graz",
      "predicate": "PARTNERED_WITH",
      "object": "Magna",
      "evidence": "..."
    }
  ]
}
```

Supported entity types are `PERSON`, `ORG`, `LOCATION`, `EVENT`, `CONCEPT`,
`PRODUCT`, `DATE`, and `OTHER`. See [examples/results.json](examples/results.json)
for a complete example.

## Additional resources

- [Corpus collection guide](docs/WP2_corpus_collection_guide.md)
- [Link-prediction notebook](notebooks/link_prediction.ipynb)
- [Example extraction output](examples/results.json)
