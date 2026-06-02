# GraphLens Prototype

Extract knowledge graphs from PDF documents with Claude, load them into Neo4j,
and run machine-learning methods (link prediction, graph embeddings) on the result.

```
PDF  →  Claude (Files API)  →  KG JSON  →  Neo4j
                                    ↓
                               KGGraph (NetworkX / PyKEEN / PyG)
                                    ↓
                           Link prediction · Node classification · …
```

---

## Project layout

```
GraphLensPrototype/
├── graphlens/               # importable library
│   ├── __init__.py
│   ├── extractor.py         # Claude-based KG extraction
│   ├── neo4j_loader.py      # Neo4j integration
│   └── ml/                  # machine-learning subpackage
│       ├── __init__.py
│       ├── graph_builder.py # KGGraph — NetworkX + int encodings + splits
│       └── link_prediction.py # HeuristicPredictor · PyKEENPredictor
├── scripts/                 # CLI entry points
│   ├── extract_kg.py        # PDF → KG JSON
│   ├── load_to_neo4j.py     # KG JSON → Neo4j
│   └── predict_links.py     # link prediction CLI
├── notebooks/
│   └── link_prediction.ipynb  # interactive exploration
├── data/
│   ├── input/               # put PDFs here  (gitignored)
│   └── output/              # extracted JSONs (gitignored)
├── examples/
│   └── results.json         # sample extraction output
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick start

### 1 — Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Core install (extraction + Neo4j + heuristic ML)
pip install -e .

# Optional: embedding-based link prediction (TransE, RotatE, …)
pip install -e ".[ml]"

# Optional: GNN methods via PyTorch Geometric
pip install -e ".[pyg]"
```

### 2 — Set environment variables

```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS / Linux
# then edit .env
```

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Get one at console.anthropic.com |
| `NEO4J_URI` | e.g. `bolt://localhost:7687` |
| `NEO4J_USER` | e.g. `neo4j` |
| `NEO4J_PASSWORD` | Your Neo4j password |

### 3 — Extract → Load → Predict

```bash
# Extract KG from a PDF
python scripts/extract_kg.py data/input/paper.pdf --out data/output/results.json

# Load into Neo4j
python scripts/load_to_neo4j.py data/output/results.json --password your-password

# Run link prediction (heuristic baseline)
python scripts/predict_links.py data/output/results.json

# Merge multiple extractions and use TransE
python scripts/predict_links.py data/output/doc1.json data/output/doc2.json \
    --model TransE --epochs 200

# Predict specific tails
python scripts/predict_links.py data/output/results.json \
    --predict-head "TU Graz" --predict-relation PARTNERED_WITH --top-k 5
```

---

## ML module

### Building a graph

```python
from graphlens.ml import KGGraph

# From a single file
kg = KGGraph.from_json("data/output/results.json")

# Merge multiple extractions (deduplicates entities and triples)
kg = KGGraph.merge([
    KGGraph.from_json("data/output/doc1.json"),
    KGGraph.from_json("data/output/doc2.json"),
])

print(kg.stats())
# {'num_entities': 10, 'num_relations': 10, 'num_triples': 10, ...}

# Train / test split (stratified by relation type)
train, test = kg.train_test_split(test_size=0.2)

# Or with a validation set
train, valid, test = kg.train_test_split(test_size=0.2, valid_size=0.1)

# Export formats
arr    = kg.to_triple_array()          # (N, 3) int32 numpy array
tf     = kg.to_pykeen_triples_factory() # requires graphlens[ml]
data   = kg.to_pyg_data()             # requires graphlens[pyg]
G_nx   = kg.graph                     # NetworkX MultiDiGraph
```

### Heuristic predictor (no ML deps)

```python
from graphlens.ml import HeuristicPredictor

predictor = HeuristicPredictor(kg, method="common_neighbors")
# methods: common_neighbors · jaccard · adamic_adar

# Evaluate (filtered ranking: MRR, Hits@1, Hits@3, Hits@10)
metrics = predictor.evaluate(test, train_triples=train)

# Predict top-k missing links for an entity
predictor.predict_tails("TU Graz", top_k=5)
# [("Magna", 3.0), ("Institute of Automotive Engineering", 1.0), ...]
```

### Embedding predictor — PyKEEN (requires `graphlens[ml]`)

```python
from graphlens.ml import PyKEENPredictor

pred = PyKEENPredictor(kg, model_name="TransE", epochs=100, embedding_dim=64)
pred.train(train, valid_triples=valid)

metrics = pred.evaluate(test)

pred.predict_tails("TU Graz", "PARTNERED_WITH", top_k=5)

pred.save("models/transe")
pred2 = PyKEENPredictor.load(kg, "models/transe")
```

Available PyKEEN models: `TransE`, `RotatE`, `DistMult`, `ComplEx`, `ConvE`, and [many more](https://pykeen.readthedocs.io/en/stable/references/models.html).

### PyTorch Geometric (requires `graphlens[pyg]`)

```python
data = kg.to_pyg_data()
# data.x           — (num_entities, num_types) one-hot node features
# data.edge_index  — (2, num_triples)
# data.edge_attr   — (num_triples,) integer relation type
```

---

## Neo4j Browser queries

```cypher
-- Full graph
MATCH (n:Entity)-[r]->(m:Entity) RETURN n, r, m

-- All persons
MATCH (n:PERSON) RETURN n

-- Ego-graph for one entity
MATCH (n:Entity {name: "TU Graz"})-[r]->(m) RETURN n, r, m
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

## Roadmap

- [ ] Batch-process a directory of PDFs in one command
- [ ] Persist `file_id` to avoid re-uploading the same document
- [ ] Node classification using entity-type labels
- [ ] Entity alignment / deduplication across documents
- [ ] Community detection (Louvain, label propagation)
- [ ] Natural-language querying via Claude + Cypher generation
- [ ] Export to RDF / Turtle, GraphML
- [ ] FastAPI service layer
