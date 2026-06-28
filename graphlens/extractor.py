"""
Claude-based knowledge-graph extraction from documents.

Typical usage
-------------
    import anthropic
    from graphlens.extractor import upload_pdf, extract, extract_from_text

    client = anthropic.Anthropic()

    # From PDF
    file_id = upload_pdf(client, Path("paper.pdf"))
    result  = extract(client, file_id, domain="academic research")

    # From scraped text — chunked by default for finer-grained recall, with a
    # verification second-pass over the full text
    result  = extract_from_text(client, text, title="...", domain="news", verify=True)

    # Disable chunking and extract from the whole text in a single call
    result  = extract_from_text(client, text, chunk_size=None)
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import anthropic
from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT_HEAD = """You are a meticulous knowledge-graph extraction expert.

Your task is to extract a **complete and exhaustive** knowledge graph from the given document.

## Entities
Extract EVERY named entity in the document — do not skip anything, even if it seems minor.
For each entity provide:
- name        : canonical surface form (resolve pronouns and abbreviations to the full name)
- type        : one of PERSON, ORG, LOCATION, EVENT, CONCEPT, PRODUCT, DATE, OTHER
- description : one sentence grounded in the document

Entity extraction rules:
- Include every named person, organisation, place, product, technology, event, and concept.
- Resolve co-references: "the university", "it", "TU Graz" → always use the canonical name.
- Split compound references: "Magna and TU Graz" → two separate entities.
- Extract explicit time anchors ("2026", "Q1 2025", "by 2030", "since 2022") as DATE
  entities whenever they qualify a trend, forecast, growth/decline statement, or
  project timeline — not for incidental dates with no analytical value (e.g. a
  citation year buried in a reference list).
- Do NOT extract bibliographic/reference-list material: cited paper titles, authors of
  cited works, journal/venue names appearing only as citations, or citation markers
  ("[12]", "Smith et al., 2020"). Only extract entities from the document's own
  substantive content.

## Relations
Extract EVERY relation between entities — both explicit and clearly implied.
For each relation provide:
- subject   : entity name (must be in the entities list)
- predicate : short label in SCREAMING_SNAKE_CASE (e.g. WORKS_FOR, FUNDED_BY, PART_OF)
- object    : entity name (must be in the entities list)
- evidence  : verbatim quote or close paraphrase from the document

Relation extraction rules:
- Check the relation vocabulary below FIRST and reuse an existing predicate whenever it
  fits — even if a more specific custom label is conceivable.
- Only invent a new SCREAMING_SNAKE_CASE predicate when none of the canonical ones apply,
  and keep it generic and reusable (e.g. SUPPLIES, not SUPPLIES_SENSORS_TO_MAGNA_2024) so
  it can recur naturally in other documents.
- Extract transitive/implicit relations when they are clearly supported by the text.
- Do NOT fabricate relations — every relation must have an evidence quote."""

_RELATION_VOCABULARY = """

## Relation vocabulary
These extractions are later merged across many documents and used to train
embedding-based link-prediction / GNN models (TransE, RotatE, PyTorch Geometric, …).
Every distinct predicate string becomes its own relation type with its own embedding.
If the same kind of relationship gets labelled with different one-off predicates across
documents ("WORKS_FOR" in one, "IS_EMPLOYED_BY" in another), the vocabulary fragments and
each relation type ends up with too few examples to learn from. Consistency across
documents matters more than squeezing out a more precise-sounding label.

Reuse one of these whenever it fits (direction is subject → object):

| Predicate         | Typical direction                                          |
|--------------------|------------------------------------------------------------|
| WORKS_FOR          | person → organization                                      |
| FOUNDED_BY         | organization → person                                       |
| MEMBER_OF          | person/organization → group or organization                |
| AFFILIATED_WITH    | person → institution                                        |
| LOCATED_IN         | entity → location                                           |
| HEADQUARTERED_IN   | organization → location                                     |
| PART_OF            | entity → larger entity (subsidiary, division, component)    |
| ACQUIRED_BY        | organization → organization                                 |
| PARTNERED_WITH     | org/person → org/person (alliance, joint project)           |
| COMPETES_WITH      | org/product → org/product                                   |
| FUNDED_BY          | entity → organization/person                                |
| SUPPLIES           | organization (supplier) → organization (customer)           |
| DEVELOPS           | organization/person → product/technology                    |
| USES               | entity → product/technology/method                          |
| PARTICIPATED_IN    | person/organization → event/project                         |

This list is a starting point, not a hard restriction — extract whatever the text
actually supports. But default to it before minting something new."""

_TREND_SIGNALS = """

## Research & market trend signals
This graph is also used to track how research focus and market activity shift over
time, by merging extractions from many documents (papers, procurement notices, news
articles, job postings) published at different dates and seeing which entities and
relations recur, grow, or fade across them. Two things make that possible:

- Anchor trend statements to a DATE entity whenever the text gives one, using a
  trend predicate from the table below. A trend statement with no date in the text
  is still worth extracting — just without the anchor.
- Deliberately look for forward-looking and comparative language, which a
  fact-only reading tends to skip: "growing demand for X", "X is gaining adoption",
  "expected to reach Y by Z", "increasing investment in X", "shift away from Y
  toward X", "X is an emerging/declining technology in [market]". These read as
  soft or qualitative rather than as a clean named-entity fact, which is exactly
  why they're easy to under-extract — but they are the signal this graph is for.
- If the document states its own publication date (or one can be inferred), extract
  it as a DATE entity and connect it to the document's main subject via
  PUBLISHED_IN — this buckets every relation from this document into a time period
  even when no other date appears anywhere else in the text.

Trend predicates (add these to the relation vocabulary above):

| Predicate          | Typical direction                                          |
|---------------------|-------------------------------------------------------------|
| GROWING_DEMAND_FOR  | market/region/organization → product/technology             |
| DECLINING_IN        | technology/product → market/region                           |
| EMERGING_IN         | technology/concept → market/region                           |
| INVESTING_IN        | organization → technology/concept/product                    |
| RESEARCHING         | organization/person → concept/technology                     |
| FOCUSING_ON         | organization → concept/technology (stated strategic focus)   |
| ADOPTING            | organization/market → product/technology                     |
| FORECASTS           | organization/report → concept (a stated prediction)          |
| PROJECTED_BY        | trend subject (technology/market/concept) → DATE              |
| PUBLISHED_IN        | document subject → DATE (anchors a document's relations to a time period) |"""

_METHOD_RESULTS_SIGNALS = """

## Research method & results signals
For research articles and technical reports, the goal is to capture what was DONE and
what was FOUND — not who is cited. Bibliographic material (cited papers, their authors,
citation markers) is out of scope (see entity rules above); spend that effort here instead.

- Extract the paper's method(s), algorithm(s), model(s), or approach(es) as CONCEPT
  entities, named as specifically as the text allows (e.g. "graph attention network",
  "k-means clustering", "ablation study").
- Extract datasets, benchmarks, or evaluation settings as CONCEPT entities.
- Extract quantitative findings as their own CONCEPT entities (e.g. "94.2% F1 score on
  SQuAD", "23% reduction in latency"), grounded in the exact figure from the text —
  do not paraphrase a number into a vaguer claim.
- The document itself (its title, or "this paper"/"this study" if no title is given)
  may be extracted as a single OTHER entity so PROPOSES/USES_METHOD can anchor to it —
  but never extract its author list as a separate entity or relation.

Method/results predicates (add these to the relation vocabulary above):

| Predicate          | Typical direction                                          |
|---------------------|-------------------------------------------------------------|
| PROPOSES            | paper → method/concept (the paper's own contribution)        |
| USES_METHOD         | paper/system/organization → method/concept (adopted, not necessarily novel) |
| EVALUATED_ON        | method → dataset/benchmark/concept                            |
| ACHIEVES            | method → result/metric (a CONCEPT entity stating the figure)  |
| OUTPERFORMS         | method → method (explicit comparative claim)                  |"""

_WORKED_EXAMPLE = """

## Worked example

Source text:
"AVL List GmbH, headquartered in Graz, Austria, develops the AVL DRIVINGCUBE™ driving
simulator. Magna International uses the system to test driver-assistance software
before road trials. AVL was founded by Hans List in 1948. In 2024, AVL partnered with
TU Graz on a joint research project; TU Graz received €2.1M in funding from the
Austrian Research Promotion Agency (FFG) for the work. Industry analysts expect
demand for vehicle-in-the-loop simulation to grow significantly across Europe by
2030, driven by tightening ADAS validation requirements."

Correct output:
```json
{
  "entities": [
    {"name": "AVL List GmbH", "type": "ORG", "description": "Austrian engineering company that develops the AVL DRIVINGCUBE driving simulator."},
    {"name": "Graz", "type": "LOCATION", "description": "City in Austria where AVL List GmbH is headquartered."},
    {"name": "Austria", "type": "LOCATION", "description": "Country containing Graz."},
    {"name": "AVL DRIVINGCUBE", "type": "PRODUCT", "description": "Driving simulator developed by AVL List GmbH."},
    {"name": "Magna International", "type": "ORG", "description": "Company that uses the AVL DRIVINGCUBE to test driver-assistance software."},
    {"name": "Hans List", "type": "PERSON", "description": "Founder of AVL List GmbH in 1948."},
    {"name": "TU Graz", "type": "ORG", "description": "University that partnered with AVL on a joint research project."},
    {"name": "Austrian Research Promotion Agency (FFG)", "type": "ORG", "description": "Agency that funded TU Graz's research project with AVL."},
    {"name": "Vehicle-in-the-Loop Simulation", "type": "CONCEPT", "description": "Simulation technique analysts expect to see growing demand for in Europe."},
    {"name": "Europe", "type": "LOCATION", "description": "Region where demand for vehicle-in-the-loop simulation is expected to grow."},
    {"name": "2030", "type": "DATE", "description": "Year by which demand for vehicle-in-the-loop simulation is projected to grow significantly."}
  ],
  "relations": [
    {"subject": "AVL List GmbH", "predicate": "HEADQUARTERED_IN", "object": "Graz", "evidence": "AVL List GmbH, headquartered in Graz, Austria"},
    {"subject": "Graz", "predicate": "LOCATED_IN", "object": "Austria", "evidence": "headquartered in Graz, Austria"},
    {"subject": "AVL List GmbH", "predicate": "DEVELOPS", "object": "AVL DRIVINGCUBE", "evidence": "develops the AVL DRIVINGCUBE™ driving simulator"},
    {"subject": "Magna International", "predicate": "USES", "object": "AVL DRIVINGCUBE", "evidence": "Magna International uses the system to test driver-assistance software"},
    {"subject": "AVL List GmbH", "predicate": "FOUNDED_BY", "object": "Hans List", "evidence": "AVL was founded by Hans List in 1948"},
    {"subject": "AVL List GmbH", "predicate": "PARTNERED_WITH", "object": "TU Graz", "evidence": "AVL partnered with TU Graz on a joint research project"},
    {"subject": "TU Graz", "predicate": "FUNDED_BY", "object": "Austrian Research Promotion Agency (FFG)", "evidence": "TU Graz received €2.1M in funding from the Austrian Research Promotion Agency (FFG) for the work"},
    {"subject": "Europe", "predicate": "GROWING_DEMAND_FOR", "object": "Vehicle-in-the-Loop Simulation", "evidence": "demand for vehicle-in-the-loop simulation to grow significantly across Europe by 2030"},
    {"subject": "Vehicle-in-the-Loop Simulation", "predicate": "PROJECTED_BY", "object": "2030", "evidence": "expect demand ... to grow significantly across Europe by 2030"}
  ]
}
```

Why these predicate choices: HEADQUARTERED_IN, LOCATED_IN, DEVELOPS, USES, FOUNDED_BY,
PARTNERED_WITH, and FUNDED_BY are all canonical predicates from the vocabulary above;
GROWING_DEMAND_FOR and PROJECTED_BY are from the trend vocabulary below — none were
invented for this specific text. Note that the forecast sentence reads as soft,
qualitative language ("industry analysts expect... to grow significantly") rather
than a clean factual statement, which is exactly the kind of sentence a fact-only
pass tends to skip — and the "2030" DATE entity is what lets this statement be
placed on a timeline once it's merged with other documents.

Common mistake to avoid: do NOT mint one-off predicates such as `IS_HEADQUARTERED_AT`,
`SUPPLIES_DRIVINGCUBE_TO`, or `RECEIVED_GRANT_FROM` for the relations above. They mean
the same thing as HEADQUARTERED_IN, SUPPLIES, and FUNDED_BY, but as new predicates
they'd each start with exactly one training example instead of adding to a shared
relation type."""

_WORKED_EXAMPLE_2 = """

## Worked example 2 — research paper methods & results

Source text:
"Document title: GraphSAGE-X: Attention-Weighted Aggregation for Graph Neural Networks

We propose GraphSAGE-X, a graph neural network that extends GraphSAGE with an
attention-weighted aggregation step. We evaluate GraphSAGE-X on the OGB-Products
benchmark, where it achieves 89.4% accuracy, outperforming the baseline GraphSAGE
model (86.1% accuracy) by 3.3 points. Related work includes Hamilton et al. [12] and
Veličković et al. [15], who introduced the original GraphSAGE and GAT architectures."

Correct output:
```json
{
  "entities": [
    {"name": "GraphSAGE-X: Attention-Weighted Aggregation for Graph Neural Networks", "type": "OTHER", "description": "The paper itself, which proposes the GraphSAGE-X architecture."},
    {"name": "GraphSAGE-X", "type": "CONCEPT", "description": "Graph neural network proposed in this paper, extending GraphSAGE with attention-weighted aggregation."},
    {"name": "GraphSAGE", "type": "CONCEPT", "description": "Baseline graph neural network architecture that GraphSAGE-X extends and outperforms."},
    {"name": "OGB-Products", "type": "CONCEPT", "description": "Benchmark dataset used to evaluate GraphSAGE-X."},
    {"name": "89.4% accuracy on OGB-Products", "type": "CONCEPT", "description": "Accuracy achieved by GraphSAGE-X on the OGB-Products benchmark."},
    {"name": "86.1% accuracy on OGB-Products", "type": "CONCEPT", "description": "Accuracy achieved by the baseline GraphSAGE model on the OGB-Products benchmark."}
  ],
  "relations": [
    {"subject": "GraphSAGE-X: Attention-Weighted Aggregation for Graph Neural Networks", "predicate": "PROPOSES", "object": "GraphSAGE-X", "evidence": "We propose GraphSAGE-X, a graph neural network that extends GraphSAGE"},
    {"subject": "GraphSAGE-X", "predicate": "EVALUATED_ON", "object": "OGB-Products", "evidence": "We evaluate GraphSAGE-X on the OGB-Products benchmark"},
    {"subject": "GraphSAGE-X", "predicate": "ACHIEVES", "object": "89.4% accuracy on OGB-Products", "evidence": "it achieves 89.4% accuracy"},
    {"subject": "GraphSAGE", "predicate": "ACHIEVES", "object": "86.1% accuracy on OGB-Products", "evidence": "the baseline GraphSAGE model (86.1% accuracy)"},
    {"subject": "GraphSAGE-X", "predicate": "OUTPERFORMS", "object": "GraphSAGE", "evidence": "outperforming the baseline GraphSAGE model (86.1% accuracy) by 3.3 points"}
  ]
}
```

Note what was deliberately left out: "Hamilton et al. [12]" and "Veličković et al. [15]"
are reference-list citations, not part of this paper's own method or results — no
entities or relations were produced for them."""

_CRITICAL_SECTION = """

## Critical
- It is far better to extract too much than to miss something important.
- Your output MUST be valid JSON that matches the schema exactly."""

_DOMAIN_SECTION = """

## Domain focus
This document belongs to the domain: **{domain}**.
Pay special attention to entities and relations that are important in this domain."""

_VERIFICATION_SYSTEM_PROMPT = (
    """You are a knowledge-graph quality reviewer.

You will be given:
1. The original document text.
2. A first-pass extraction (entities and relations already found).

Your job is to identify **only the entities and relations that were MISSED** in the first pass.
Do not repeat anything already in the first-pass result.
Apply the same exhaustive extraction rules as the original pass.
Reuse the same predicates the first pass already used, and the canonical vocabulary below,
rather than inventing synonyms for relations that are conceptually the same.
Pay particular attention to forward-looking or comparative trend language (growth,
decline, forecasts, emerging/declining technology, research focus, investment), and to
a paper's own methodology and quantitative results, both of which a first pass commonly
under-extracts because they read as narrative rather than as a clean named-entity fact.
Do not flag missing citation/reference-list material (cited papers, their authors, or
citation markers) — that is out of scope by design, not a miss."""
    + _RELATION_VOCABULARY
    + _TREND_SIGNALS
    + _METHOD_RESULTS_SIGNALS
    + """

Return a JSON object with the same schema: {"entities": [...], "relations": [...]}
containing ONLY the additions — an empty list is fine if nothing was missed."""
)

# ---------------------------------------------------------------------------
# Output schema (shared by all extraction calls)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": "string"},
                    "type":        {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject":   {"type": "string"},
                    "predicate": {"type": "string"},
                    "object":    {"type": "string"},
                    "evidence":  {"type": "string"},
                },
                "required": ["subject", "predicate", "object", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(domain: str | None) -> str:
    domain_section = (
        _DOMAIN_SECTION.format(domain=domain) if domain else ""
    )
    return (
        _BASE_SYSTEM_PROMPT_HEAD
        + _RELATION_VOCABULARY
        + _TREND_SIGNALS
        + _METHOD_RESULTS_SIGNALS
        + _WORKED_EXAMPLE
        + _WORKED_EXAMPLE_2
        + _CRITICAL_SECTION
        + domain_section
    )


def _stream_extract(
    client:         anthropic.Anthropic,
    messages:       list[dict],
    system_prompt:  str,
    use_files_beta: bool = False,
) -> dict:
    """Run a single streaming extraction call and return the parsed JSON dict."""
    stream_kwargs = dict(
        model="claude-opus-4-8",
        max_tokens=32000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": OUTPUT_SCHEMA,
            }
        },
    )

    if use_files_beta:
        with client.beta.messages.stream(
            **stream_kwargs, betas=["files-api-2025-04-14"]
        ) as stream:
            message = stream.get_final_message()
    else:
        with client.messages.stream(**stream_kwargs) as stream:
            message = stream.get_final_message()

    u = message.usage
    print(
        f"  Tokens — input: {u.input_tokens}, output: {u.output_tokens}, "
        f"cache_read: {getattr(u, 'cache_read_input_tokens', 0)}, "
        f"cache_write: {getattr(u, 'cache_creation_input_tokens', 0)}",
        file=sys.stderr,
    )

    text_block = next(b for b in message.content if b.type == "text")
    return json.loads(text_block.text)


def _merge(base: dict, additions: dict) -> dict:
    """Merge *additions* into *base*, deduplicating on entity name and triple key."""
    existing_names = {e["name"] for e in base["entities"]}
    for entity in additions.get("entities", []):
        if entity["name"] not in existing_names:
            base["entities"].append(entity)
            existing_names.add(entity["name"])

    existing_triples = {
        (r["subject"], r["predicate"], r["object"])
        for r in base["relations"]
    }
    for rel in additions.get("relations", []):
        key = (rel["subject"], rel["predicate"], rel["object"])
        if key not in existing_triples:
            base["relations"].append(rel)
            existing_triples.add(key)

    return base


# ---------------------------------------------------------------------------
# Chunking (plain-text extraction only)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE    = 12000  # characters, ≈ 3k tokens of body text
DEFAULT_CHUNK_OVERLAP = 600     # characters of trailing context carried forward


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap:    int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into ~chunk_size-character pieces for finer-grained extraction.

    Breaks at the nearest paragraph boundary (or failing that, whitespace) at or
    before the target cut point, so words and sentences aren't split mid-way.
    Each chunk after the first is prefixed with *overlap* characters from the
    end of the previous one, so entities/relations straddling a cut aren't lost.
    """
    text = text.strip()
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _extract_chunk(
    client:        anthropic.Anthropic,
    system_prompt: str,
    chunk:         str,
    title:         str,
    index:         int,
    total:         int,
    known_entities: list[str],
) -> dict:
    """Run a single extraction call on one chunk of a larger document."""
    header = f"Document title: {title}\n\n" if title else ""
    if total > 1:
        header += (
            f"This is excerpt {index} of {total} from a larger document, in reading "
            "order. It may begin or end mid-sentence — extract whatever entities and "
            "relations this excerpt supports regardless.\n\n"
        )
        if known_entities:
            header += (
                "Entities already extracted from earlier excerpts of this document — "
                "reuse these exact names if this excerpt refers to the same entity "
                "(e.g. by pronoun or abbreviation) instead of minting a new one:\n"
                + ", ".join(known_entities) + "\n\n"
            )

    user_message = (
        f"{header}{chunk}\n\n"
        "---\n"
        "Extract ALL entities and relations from the text above. "
        "Be exhaustive — do not skip anything. "
        "Return strictly valid JSON matching the schema."
    )
    return _stream_extract(client, [{"role": "user", "content": user_message}], system_prompt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_pdf(client: anthropic.Anthropic, pdf_path: Path) -> str:
    """Upload *pdf_path* to the Files API and return the remote ``file_id``.

    The caller is responsible for deleting the file when done::

        client.beta.files.delete(file_id)
    """
    print(f"Uploading {pdf_path.name} …", file=sys.stderr)
    with pdf_path.open("rb") as fh:
        result = client.beta.files.upload(
            file=(pdf_path.name, fh, "application/pdf"),
        )
    print(f"  → file_id: {result.id}", file=sys.stderr)
    return result.id


def _extract_pdf_chunk(
    client:         anthropic.Anthropic,
    system_prompt:  str,
    file_id:        str,
    index:          int,
    total:          int,
    page_start:     int,
    page_end:       int,
    known_entities: list[str],
) -> dict:
    """Run a single extraction call on one page-range chunk of a larger PDF."""
    note = (
        f"This excerpt contains pages {page_start}-{page_end} ({index} of {total} "
        "chunks) of a larger document, in reading order. It may begin or end "
        "mid-section — extract whatever entities and relations it supports regardless."
    )
    if known_entities:
        note += (
            "\n\nEntities already extracted from earlier chunks of this document — "
            "reuse these exact names if this excerpt refers to the same entity "
            "(e.g. by pronoun or abbreviation) instead of minting a new one:\n"
            + ", ".join(known_entities)
        )

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                    "title": "Document chunk for knowledge-graph extraction",
                },
                {
                    "type": "text",
                    "text": (
                        f"{note}\n\n"
                        "Extract ALL entities and relations from the document chunk "
                        "above. Be exhaustive — do not skip anything. "
                        "Return strictly valid JSON matching the schema."
                    ),
                },
            ],
        }
    ]
    return _stream_extract(client, messages, system_prompt, use_files_beta=True)


def _extract_pdf_chunked(
    client:          anthropic.Anthropic,
    pdf_path:        Path,
    system_prompt:   str,
    pages_per_chunk: int,
    chunk_dir:       str | Path | None,
) -> dict:
    """Split *pdf_path* into page-range chunks and extract from each separately."""
    reader   = PdfReader(str(pdf_path))
    n_pages  = len(reader.pages)
    n_chunks = -(-n_pages // pages_per_chunk)  # ceil division
    width    = len(str(n_chunks))

    save_dir = Path(chunk_dir) if chunk_dir else None
    work_dir = save_dir or Path(tempfile.mkdtemp(prefix="graphlens_chunks_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Extracting from PDF ({n_pages} pages, {n_chunks} chunks of "
        f"{pages_per_chunk} pages) …",
        file=sys.stderr,
    )

    result: dict = {"entities": [], "relations": []}
    try:
        for i, start in enumerate(range(0, n_pages, pages_per_chunk), start=1):
            end  = min(start + pages_per_chunk, n_pages)
            stem = f"chunk_{i:0{width}d}"

            writer = PdfWriter()
            for page in reader.pages[start:end]:
                writer.add_page(page)
            chunk_pdf_path = work_dir / f"{stem}.pdf"
            with chunk_pdf_path.open("wb") as fh:
                writer.write(fh)

            print(f"  Chunk {i}/{n_chunks} (pages {start + 1}-{end}) …", file=sys.stderr)
            chunk_file_id = upload_pdf(client, chunk_pdf_path)
            try:
                known_entities = [e["name"] for e in result["entities"]]
                addition = _extract_pdf_chunk(
                    client, system_prompt, chunk_file_id,
                    i, n_chunks, start + 1, end, known_entities,
                )
            finally:
                try:
                    client.beta.files.delete(chunk_file_id)
                except Exception as exc:
                    print(
                        f"  Warning: could not delete chunk file {chunk_file_id}: {exc}",
                        file=sys.stderr,
                    )

            if save_dir:
                (work_dir / f"{stem}_extraction.txt").write_text(
                    format_result(addition), encoding="utf-8"
                )

            result = _merge(result, addition)
    finally:
        if not save_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


def extract(
    client:  anthropic.Anthropic,
    file_id: str,
    *,
    domain:      str | None = None,
    verify:      bool = False,
    pdf_path:    str | Path | None = None,
    chunk_pages: int | None = None,
    chunk_dir:   str | Path | None = None,
) -> dict:
    """Run KG extraction on an already-uploaded PDF document.

    Parameters
    ----------
    client:      Anthropic API client.
    file_id:     ID returned by :func:`upload_pdf` for the whole document. Used
                 directly when *chunk_pages* is ``None``; otherwise only used
                 for the optional verification pass.
    domain:      Optional domain hint (e.g. ``"academic research"``, ``"news"``,
                 ``"automotive engineering"``).  Helps the model focus on what
                 matters in that domain.
    verify:      If ``True``, run a second Claude pass over the whole document
                 to find entities and relations missed in the first pass(es),
                 then merge the results. Increases recall at the cost of
                 roughly 2× token usage.
    pdf_path:    Local path to the source PDF. Required when *chunk_pages* is
                 set, since the file must be re-split and re-uploaded per chunk.
    chunk_pages: Split the PDF into pieces of this many pages and extract from
                 each separately — smaller pieces get the model's full
                 attention, which improves recall of fine-grained details.
                 Pass ``None`` (default) to extract from the whole document in
                 a single call.
    chunk_dir:   If given (with *chunk_pages*), save each page-range chunk as
                 ``chunk_NNN.pdf`` and its extraction as
                 ``chunk_NNN_extraction.txt`` in this directory.
    """
    system_prompt = _build_system_prompt(domain)

    if chunk_pages:
        if not pdf_path:
            raise ValueError("chunk_pages requires pdf_path (the local PDF file).")
        result = _extract_pdf_chunked(
            client, Path(pdf_path), system_prompt, chunk_pages, chunk_dir
        )
    else:
        print(
            "Extracting entities and relations (pass 1) …"
            + (f"  [domain: {domain}]" if domain else ""),
            file=sys.stderr,
        )

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id},
                        "title": "Document for knowledge-graph extraction",
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL entities and relations from the document above. "
                            "Be exhaustive — do not skip anything. "
                            "Return strictly valid JSON matching the schema."
                        ),
                    },
                ],
            }
        ]

        result = _stream_extract(
            client, messages, system_prompt, use_files_beta=True
        )

    if verify:
        result = _verify_pass_pdf(client, file_id, result)

    return result


def extract_from_text(
    client: anthropic.Anthropic,
    text:   str,
    title:  str = "",
    *,
    domain:        str | None = None,
    verify:        bool = False,
    chunk_size:    int | None = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_dir:     str | Path | None = None,
) -> dict:
    """Run KG extraction on a plain-text string (e.g. scraped web content).

    Parameters
    ----------
    client:        Anthropic API client.
    text:          The document text to extract from.
    title:         Optional document title included as context for the model.
    domain:        Optional domain hint (e.g. ``"news"``, ``"academic research"``).
    verify:        If ``True``, run a second pass over the full text to find
                   anything missed across all chunks, then merge.
    chunk_size:    Split *text* into pieces of roughly this many characters and
                   extract from each separately — smaller pieces get the model's
                   full attention, which improves recall of fine-grained details.
                   Pass ``None`` to extract from the whole text in a single call.
    chunk_overlap: Characters of trailing context carried from one chunk into
                   the next, so entities/relations spanning a chunk boundary
                   aren't lost.
    chunk_dir:     If given, write each chunk's text to ``chunk_NNN.txt`` and its
                   extracted entities/relations to ``chunk_NNN_extraction.txt``
                   in this directory (created if needed) — useful for inspecting
                   what each chunk actually contributed.
    """
    system_prompt = _build_system_prompt(domain)
    chunks = _chunk_text(text, chunk_size, chunk_overlap) if chunk_size else [text]

    print(
        f"Extracting from text ({len(text):,} chars"
        + (f", {len(chunks)} chunks" if len(chunks) > 1 else "")
        + ")"
        + (f" — {title!r}" if title else "")
        + (f"  [domain: {domain}]" if domain else "")
        + " …",
        file=sys.stderr,
    )

    chunk_path = Path(chunk_dir) if chunk_dir else None
    if chunk_path:
        chunk_path.mkdir(parents=True, exist_ok=True)
    width = len(str(len(chunks)))

    result: dict = {"entities": [], "relations": []}
    for i, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"  Chunk {i}/{len(chunks)} ({len(chunk):,} chars) …", file=sys.stderr)
        known_entities = [e["name"] for e in result["entities"]]
        addition = _extract_chunk(
            client, system_prompt, chunk, title, i, len(chunks), known_entities
        )

        if chunk_path:
            stem = f"chunk_{i:0{width}d}"
            (chunk_path / f"{stem}.txt").write_text(chunk, encoding="utf-8")
            (chunk_path / f"{stem}_extraction.txt").write_text(
                format_result(addition), encoding="utf-8"
            )
            print(f"    saved {stem}.txt / {stem}_extraction.txt", file=sys.stderr)

        result = _merge(result, addition)

    if verify:
        result = _verify_pass_text(client, text, title, result)

    return result


# ---------------------------------------------------------------------------
# Verification pass helpers
# ---------------------------------------------------------------------------

def _verify_pass_pdf(
    client:       anthropic.Anthropic,
    file_id:      str,
    first_result: dict,
) -> dict:
    """Second pass: find what was missed in *first_result* for a PDF document."""
    print("Verification pass (pass 2) …", file=sys.stderr)

    first_json = json.dumps(first_result, indent=2)
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                    "title": "Original document",
                },
                {
                    "type": "text",
                    "text": (
                        "First-pass extraction result:\n"
                        f"```json\n{first_json}\n```\n\n"
                        "Review the original document above and return ONLY the entities "
                        "and relations that were missed. Return an empty list if nothing "
                        "was missed."
                    ),
                },
            ],
        }
    ]

    additions = _stream_extract(
        client, messages, _VERIFICATION_SYSTEM_PROMPT, use_files_beta=True
    )
    merged = _merge(first_result, additions)

    added_e = len(merged["entities"])  - len(first_result["entities"])
    added_r = len(merged["relations"]) - len(first_result["relations"])
    print(
        f"  Verification added {added_e} entities and {added_r} relations.",
        file=sys.stderr,
    )
    return merged


def _verify_pass_text(
    client:       anthropic.Anthropic,
    text:         str,
    title:        str,
    first_result: dict,
) -> dict:
    """Second pass: find what was missed in *first_result* for plain text."""
    print("Verification pass (pass 2) …", file=sys.stderr)

    header     = f"Document title: {title}\n\n" if title else ""
    first_json = json.dumps(first_result, indent=2)
    user_message = (
        f"{header}{text}\n\n"
        "---\n"
        "First-pass extraction result:\n"
        f"```json\n{first_json}\n```\n\n"
        "Review the original text above and return ONLY the entities and relations "
        "that were missed. Return an empty list if nothing was missed."
    )

    additions = _stream_extract(
        client,
        [{"role": "user", "content": user_message}],
        _VERIFICATION_SYSTEM_PROMPT,
    )
    merged = _merge(first_result, additions)

    added_e = len(merged["entities"])  - len(first_result["entities"])
    added_r = len(merged["relations"]) - len(first_result["relations"])
    print(
        f"  Verification added {added_e} entities and {added_r} relations.",
        file=sys.stderr,
    )
    return merged


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_result(result: dict) -> str:
    """Render *result* as the human-readable entities/relations summary."""
    entities  = result.get("entities", [])
    relations = result.get("relations", [])

    lines = [f"\n{'=' * 60}", f"ENTITIES  ({len(entities)})", "=" * 60]
    for e in entities:
        lines.append(f"  [{e['type']}] {e['name']}")
        lines.append(f"    {e['description']}")

    lines += [f"\n{'=' * 60}", f"RELATIONS ({len(relations)})", "=" * 60]
    for r in relations:
        lines.append(f"  {r['subject']}  —[{r['predicate']}]→  {r['object']}")
        lines.append(f"    evidence: {r['evidence']}")
    lines.append("")

    return "\n".join(lines)


def pretty_print(result: dict) -> None:
    """Print a human-readable summary of extracted entities and relations."""
    print(format_result(result))


_ENTITY_LINE_RE   = re.compile(r"^  \[(?P<type>[^\]]+)\] (?P<name>.+)$")
_RELATION_LINE_RE = re.compile(r"^  (?P<subject>.+?)  —\[(?P<predicate>[^\]]+)\]→  (?P<object>.+)$")


def parse_extraction_text(text: str) -> dict:
    """Parse the output of :func:`format_result` back into entities/relations.

    Inverse of :func:`format_result` — used to reconstruct a JSON result from
    saved ``chunk_NNN_extraction.txt`` files (e.g. after a partial run).
    """
    lines = text.splitlines()
    entities:  list[dict] = []
    relations: list[dict] = []

    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ENTITIES"):
            section = "entities"
        elif line.startswith("RELATIONS"):
            section = "relations"
        elif section == "entities":
            m = _ENTITY_LINE_RE.match(line)
            if m and i + 1 < len(lines):
                entities.append({
                    "name":        m.group("name").strip(),
                    "type":        m.group("type").strip(),
                    "description": lines[i + 1].strip(),
                })
                i += 1
        elif section == "relations":
            m = _RELATION_LINE_RE.match(line)
            if m and i + 1 < len(lines):
                evidence = lines[i + 1].strip()
                if evidence.lower().startswith("evidence:"):
                    evidence = evidence[len("evidence:"):].strip()
                relations.append({
                    "subject":   m.group("subject").strip(),
                    "predicate": m.group("predicate").strip(),
                    "object":    m.group("object").strip(),
                    "evidence":  evidence,
                })
                i += 1
        i += 1

    return {"entities": entities, "relations": relations}


def combine_chunk_extractions(chunk_dir: str | Path) -> dict:
    """Combine every ``*_extraction.txt`` file in *chunk_dir* into one
    deduplicated ``{"entities": [...], "relations": [...]}`` dict.

    Reconstructs a final result from saved chunk extractions — e.g. when a
    chunked run was interrupted before producing a merged JSON output, or to
    recombine chunks independently of the original extraction run.
    """
    chunk_dir = Path(chunk_dir)
    paths = sorted(chunk_dir.glob("*_extraction.txt"))
    if not paths:
        raise FileNotFoundError(f"No *_extraction.txt files found in {chunk_dir}")

    result: dict = {"entities": [], "relations": []}
    for path in paths:
        addition = parse_extraction_text(path.read_text(encoding="utf-8"))
        result = _merge(result, addition)
    return result
