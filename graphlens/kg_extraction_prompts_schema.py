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
- Canonicalize surface-form variants: if the same real-world entity appears under
  multiple surface forms in the document (e.g. "AVL", "AVL List GmbH", "the company"),
  pick the single most complete/official form as the canonical name and use that
  exact string everywhere — in the entities list and in every relation that
  references it. Never let the same entity appear twice under different names.
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
- subject         : entity name (must be in the entities list)
- predicate       : short label in SCREAMING_SNAKE_CASE (e.g. WORKS_FOR, FUNDED_BY, PART_OF)
- object          : entity name (must be in the entities list)
- source_sentence : ONE single, contiguous sentence from the document — verbatim, or a
                     close paraphrase of that one sentence — that states the relation.
                     This is the passage-level provenance for the relation, so it must be
                     locatable as a real span of the document text. Never splice together
                     fragments from two different sentences (e.g. with "...") into one
                     source_sentence, and never summarize across multiple sentences — if
                     several sentences together support the relation, pick the single one
                     that most directly and completely supports it on its own.
- confidence      : your own confidence that this relation is correctly and precisely
                     supported by source_sentence, from 0.0 to 1.0. Use roughly:
                     0.9-1.0 for an explicit, unambiguous statement; 0.6-0.85 for a
                     relation that requires a small amount of paraphrase or co-reference
                     resolution; 0.3-0.55 for a relation that is only weakly or
                     indirectly implied. Never output a relation you would rate below 0.3
                     — if it's that uncertain, leave it out instead.

Relation extraction rules:
- Check the relation vocabulary below FIRST and reuse an existing predicate whenever it
  fits — even if a more specific custom label is conceivable.
- Only invent a new SCREAMING_SNAKE_CASE predicate when none of the canonical ones apply,
  and keep it generic and reusable (e.g. SUPPLIES, not SUPPLIES_SENSORS_TO_MAGNA_2024) so
  it can recur naturally in other documents.
- Extract transitive/implicit relations when they are clearly supported by the text.
- Do NOT fabricate relations — every relation must have a source_sentence quote.
- De-duplicate: if the same (subject, predicate, object) triple is supported by more
  than one sentence, emit it only once. Prefer the clearest or most specific sentence
  as source_sentence; do not repeat the triple with different source_sentence strings."""

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
  Note: these figure-specific entities are usually unique to this document and are not
  expected to recur or merge across the corpus the way an org or method name would.
  Extract them anyway for provenance and for the ACHIEVES relation below, but do not
  treat their low merge rate as a bug — it's an expected property of document-specific
  results, unlike methods/datasets/technologies, which should recur and are the ones
  the downstream link-prediction model relies on.
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
    {"subject": "AVL List GmbH", "predicate": "HEADQUARTERED_IN", "object": "Graz", "source_sentence": "AVL List GmbH, headquartered in Graz, Austria, develops the AVL DRIVINGCUBE™ driving simulator.", "confidence": 0.98},
    {"subject": "Graz", "predicate": "LOCATED_IN", "object": "Austria", "source_sentence": "AVL List GmbH, headquartered in Graz, Austria, develops the AVL DRIVINGCUBE™ driving simulator.", "confidence": 0.9},
    {"subject": "AVL List GmbH", "predicate": "DEVELOPS", "object": "AVL DRIVINGCUBE", "source_sentence": "AVL List GmbH, headquartered in Graz, Austria, develops the AVL DRIVINGCUBE™ driving simulator.", "confidence": 0.98},
    {"subject": "Magna International", "predicate": "USES", "object": "AVL DRIVINGCUBE", "source_sentence": "Magna International uses the system to test driver-assistance software before road trials.", "confidence": 0.95},
    {"subject": "AVL List GmbH", "predicate": "FOUNDED_BY", "object": "Hans List", "source_sentence": "AVL was founded by Hans List in 1948.", "confidence": 0.98},
    {"subject": "AVL List GmbH", "predicate": "PARTNERED_WITH", "object": "TU Graz", "source_sentence": "In 2024, AVL partnered with TU Graz on a joint research project.", "confidence": 0.95},
    {"subject": "TU Graz", "predicate": "FUNDED_BY", "object": "Austrian Research Promotion Agency (FFG)", "source_sentence": "TU Graz received €2.1M in funding from the Austrian Research Promotion Agency (FFG) for the work.", "confidence": 0.95},
    {"subject": "Europe", "predicate": "GROWING_DEMAND_FOR", "object": "Vehicle-in-the-Loop Simulation", "source_sentence": "Industry analysts expect demand for vehicle-in-the-loop simulation to grow significantly across Europe by 2030, driven by tightening ADAS validation requirements.", "confidence": 0.75},
    {"subject": "Vehicle-in-the-Loop Simulation", "predicate": "PROJECTED_BY", "object": "2030", "source_sentence": "Industry analysts expect demand for vehicle-in-the-loop simulation to grow significantly across Europe by 2030, driven by tightening ADAS validation requirements.", "confidence": 0.75}
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
    {"subject": "GraphSAGE-X: Attention-Weighted Aggregation for Graph Neural Networks", "predicate": "PROPOSES", "object": "GraphSAGE-X", "source_sentence": "We propose GraphSAGE-X, a graph neural network that extends GraphSAGE with an attention-weighted aggregation step.", "confidence": 0.97},
    {"subject": "GraphSAGE-X", "predicate": "EVALUATED_ON", "object": "OGB-Products", "source_sentence": "We evaluate GraphSAGE-X on the OGB-Products benchmark, where it achieves 89.4% accuracy, outperforming the baseline GraphSAGE model (86.1% accuracy) by 3.3 points.", "confidence": 0.95},
    {"subject": "GraphSAGE-X", "predicate": "ACHIEVES", "object": "89.4% accuracy on OGB-Products", "source_sentence": "We evaluate GraphSAGE-X on the OGB-Products benchmark, where it achieves 89.4% accuracy, outperforming the baseline GraphSAGE model (86.1% accuracy) by 3.3 points.", "confidence": 0.97},
    {"subject": "GraphSAGE", "predicate": "ACHIEVES", "object": "86.1% accuracy on OGB-Products", "source_sentence": "We evaluate GraphSAGE-X on the OGB-Products benchmark, where it achieves 89.4% accuracy, outperforming the baseline GraphSAGE model (86.1% accuracy) by 3.3 points.", "confidence": 0.95},
    {"subject": "GraphSAGE-X", "predicate": "OUTPERFORMS", "object": "GraphSAGE", "source_sentence": "We evaluate GraphSAGE-X on the OGB-Products benchmark, where it achieves 89.4% accuracy, outperforming the baseline GraphSAGE model (86.1% accuracy) by 3.3 points.", "confidence": 0.95}
  ]
}
```

Note what was deliberately left out: "Hamilton et al. [12]" and "Veličković et al. [15]"
are reference-list citations, not part of this paper's own method or results — no
entities or relations were produced for them."""

_RESEARCH_GAP_SIGNALS = """

## Research gap extraction
Beyond entities and relations, extract the paper's own stated or clearly implied
research gaps — this is the highest-value signal for surfacing insights across a
corpus of recent publications, so treat it as a first-class extraction target, not
an afterthought.

Look for:
- Explicit gap language: "remains an open problem", "has not yet been explored",
  "future work should investigate", "is left for future research", "no existing
  method addresses", "further investigation is needed", "to the best of our
  knowledge, no prior work...".
- Stated limitations of the paper's own method: "our approach does not scale to
  X", "this study is limited to Y", "we do not address Z".
- Conflicting or unresolved findings the paper flags relative to prior work
  ("in contrast to [prior claim], we find...", "these results are inconsistent
  with...").
- Missing datasets, benchmarks, or evaluation settings the paper explicitly calls
  out as absent from the field.

Do NOT infer a gap the document doesn't state or clearly imply — a gap must be
grounded in the text the same way a relation must have evidence. Do not manufacture
a gap just to have one; an empty research_gaps list is the correct output for a
document that states none.

For each gap provide:
- description         : the gap itself, in your own words, one to two sentences
- gap_type            : one of UNADDRESSED_LIMITATION, CONFLICTING_FINDINGS,
                         METHODOLOGICAL_GAP, MISSING_BENCHMARK_OR_DATASET,
                         UNDEREXPLORED_APPLICATION, SCALABILITY_OR_GENERALIZATION_GAP,
                         OTHER
- related_concepts    : list of entity names (must already appear in the entities
                         list) that the gap concerns — typically the method,
                         dataset, or CONCEPT entities involved
- evidence            : verbatim quote or close paraphrase from the document
- suggested_direction : optional — only fill this in if the document itself
                         proposes or hints at a direction; leave it as an empty
                         string if the document only names the gap without
                         suggesting how to close it. Do not invent a research
                         direction the document doesn't suggest.

## Worked example 3 -- research gap

Source text (continuing from Worked Example 2):
"While GraphSAGE-X improves accuracy on static graphs, it assumes a fixed graph
structure at inference time. Extending attention-weighted aggregation to dynamic,
evolving graphs remains an open problem and is left for future work."

Correct additional output:
```json
{
  "research_gaps": [
    {
      "description": "GraphSAGE-X assumes a fixed graph structure and does not handle dynamic or evolving graphs at inference time.",
      "gap_type": "SCALABILITY_OR_GENERALIZATION_GAP",
      "related_concepts": ["GraphSAGE-X"],
      "evidence": "Extending attention-weighted aggregation to dynamic, evolving graphs remains an open problem and is left for future work.",
      "suggested_direction": "Extend attention-weighted aggregation to dynamic, evolving graphs."
    }
  ]
}
```

Note `suggested_direction` here is filled in because the source text itself names
the direction ("extending... to dynamic, evolving graphs") -- it is not Claude's own
speculation."""

_CRITICAL_SECTION = """

## Critical
- It is far better to extract too much than to miss something important.
- Your output MUST be valid JSON that matches the schema exactly."""

_DOMAIN_SECTION = """

## Domain focus
This document belongs to the domain: **{domain}**.
Pay special attention to entities and relations that are important in this domain."""


def build_verification_system_prompt(domain: str | None = None) -> str:
    """Build the verification/review-pass system prompt.

    Accepts the same `domain` used for the first pass so the reviewer applies the
    same domain focus when looking for missed entities/relations, rather than
    reviewing generically.
    """
    base = (
        """You are a knowledge-graph quality reviewer.

You will be given:
1. The original document text.
2. A first-pass extraction (entities and relations already found).

Your job is to identify **only the entities and relations that were MISSED** in the first pass.
Do not repeat anything already in the first-pass result.
Apply the same exhaustive extraction rules as the original pass, including
canonicalizing surface-form variants to match names already used in the first pass
(do not introduce a second name for an entity the first pass already extracted).
Reuse the same predicates the first pass already used, and the canonical vocabulary below,
rather than inventing synonyms for relations that are conceptually the same.
Pay particular attention to forward-looking or comparative trend language (growth,
decline, forecasts, emerging/declining technology, research focus, investment), and to
a paper's own methodology and quantitative results, both of which a first pass commonly
under-extracts because they read as narrative rather than as a clean named-entity fact.
Do not flag missing citation/reference-list material (cited papers, their authors, or
citation markers) — that is out of scope by design, not a miss.
Before including a relation, check it is not already present in the first pass under
the same (subject, predicate, object) triple, even if the source_sentence differs.
Apply the same standard to research gaps: only report a gap if it is not already
present in the first pass, and never infer one the document doesn't state or
clearly imply."""
        + _RELATION_VOCABULARY
        + _TREND_SIGNALS
        + _METHOD_RESULTS_SIGNALS
        + _RESEARCH_GAP_SIGNALS
    )
    if domain:
        base += _DOMAIN_SECTION.format(domain=domain)
    base += """

Return a JSON object with the same schema: {"entities": [...], "relations": [...],
"research_gaps": [...]} containing ONLY the additions — empty lists are fine if
nothing was missed."""
    return base


def build_extraction_system_prompt(domain: str | None = None) -> str:
    """Build the first-pass extraction system prompt.

    Composes all sections in order, including the research-gap signals, so the
    first pass and the verification pass apply identical rules for what counts
    as a gap.
    """
    prompt = (
        _BASE_SYSTEM_PROMPT_HEAD
        + _RELATION_VOCABULARY
        + _TREND_SIGNALS
        + _METHOD_RESULTS_SIGNALS
        + _RESEARCH_GAP_SIGNALS
        + _WORKED_EXAMPLE
        + _WORKED_EXAMPLE_2
        + _CRITICAL_SECTION
    )
    if domain:
        prompt += _DOMAIN_SECTION.format(domain=domain)
    return prompt


# JSON Schema for the Anthropic API's tool-use / structured-output mode.
# Pass this as the `input_schema` of a tool and set `tool_choice` to force the
# model to call it, so the response shape is validated by the API rather than
# hoped for from prose + examples.
EXTRACTION_TOOL_SCHEMA = {
    "name": "record_knowledge_graph_extraction",
    "description": (
        "Record the entities, relations, and research gaps extracted from the "
        "document, following the extraction rules in the system prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "Every named entity extracted from the document.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Canonical surface form of the entity.",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "PERSON",
                                "ORG",
                                "LOCATION",
                                "EVENT",
                                "CONCEPT",
                                "PRODUCT",
                                "DATE",
                                "OTHER",
                            ],
                        },
                        "description": {
                            "type": "string",
                            "description": "One sentence grounded in the document.",
                        },
                    },
                    "required": ["name", "type", "description"],
                    "additionalProperties": False,
                },
            },
            "relations": {
                "type": "array",
                "description": "Every relation extracted between two entities.",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Entity name; must appear in `entities`.",
                        },
                        "predicate": {
                            "type": "string",
                            "pattern": "^[A-Z][A-Z0-9_]*$",
                            "description": (
                                "SCREAMING_SNAKE_CASE label. Reuse the canonical "
                                "vocabulary from the system prompt whenever it fits."
                            ),
                        },
                        "object": {
                            "type": "string",
                            "description": "Entity name; must appear in `entities`.",
                        },
                        "source_sentence": {
                            "type": "string",
                            "description": (
                                "Verbatim sentence (or close paraphrase) from the "
                                "document stating the relation. Passage-level "
                                "provenance for this relation."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Self-assessed confidence that source_sentence "
                                "precisely supports this relation."
                            ),
                        },
                    },
                    "required": ["subject", "predicate", "object", "source_sentence", "confidence"],
                    "additionalProperties": False,
                },
            },
            "research_gaps": {
                "type": "array",
                "description": (
                    "Research gaps the document itself states or clearly implies. "
                    "Empty list if none are stated — do not infer gaps."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "The gap, in one to two sentences.",
                        },
                        "gap_type": {
                            "type": "string",
                            "enum": [
                                "UNADDRESSED_LIMITATION",
                                "CONFLICTING_FINDINGS",
                                "METHODOLOGICAL_GAP",
                                "MISSING_BENCHMARK_OR_DATASET",
                                "UNDEREXPLORED_APPLICATION",
                                "SCALABILITY_OR_GENERALIZATION_GAP",
                                "OTHER",
                            ],
                        },
                        "related_concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Entity names (must appear in `entities`) that this "
                                "gap concerns."
                            ),
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Verbatim quote or close paraphrase.",
                        },
                        "suggested_direction": {
                            "type": "string",
                            "description": (
                                "Direction the document itself proposes for closing "
                                "the gap. Empty string if the document names the gap "
                                "without suggesting a direction — never invent one."
                            ),
                        },
                    },
                    "required": [
                        "description",
                        "gap_type",
                        "related_concepts",
                        "evidence",
                        "suggested_direction",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities", "relations", "research_gaps"],
        "additionalProperties": False,
    },
}


# Backward-compatible module-level constant (no domain applied).
# Prefer calling build_verification_system_prompt(domain=...) directly when a
# domain is known, so the reviewer pass gets the same focus as the first pass.
_VERIFICATION_SYSTEM_PROMPT = build_verification_system_prompt()


# ---------------------------------------------------------------------------
# Example usage: forcing structured output via tool_choice.
#
# `tool_choice={"type": "tool", "name": ...}` forces the model to call this
# specific tool, so `response.content[0].input` is guaranteed to validate
# against EXTRACTION_TOOL_SCHEMA — no prose to strip, no free-form JSON to
# parse defensively. This is the piece that removes the malformed-output
# failure class entirely; the prose sections above still do the substantive
# work of teaching the model *what* to put in each field.
# ---------------------------------------------------------------------------
def extract_knowledge_graph(client, document_text: str, domain: str | None = None):
    """Run the first-pass extraction against the Anthropic API with the schema enforced.

    `client` is an `anthropic.Anthropic()` instance. Returns the parsed dict
    matching EXTRACTION_TOOL_SCHEMA's input_schema.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=build_extraction_system_prompt(domain=domain),
        tools=[EXTRACTION_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": EXTRACTION_TOOL_SCHEMA["name"]},
        messages=[{"role": "user", "content": document_text}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == EXTRACTION_TOOL_SCHEMA["name"]:
            return block.input  # already a validated dict, no json.loads needed
    raise ValueError("Model did not call the extraction tool.")


def verify_knowledge_graph_extraction(
    client, document_text: str, first_pass: dict, domain: str | None = None
):
    """Run the verification/review pass, also schema-enforced. Returns only additions."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=build_verification_system_prompt(domain=domain),
        tools=[EXTRACTION_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": EXTRACTION_TOOL_SCHEMA["name"]},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original document:\n{document_text}\n\n"
                    f"First-pass extraction:\n{json.dumps(first_pass)}"
                ),
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == EXTRACTION_TOOL_SCHEMA["name"]:
            return block.input
    raise ValueError("Model did not call the extraction tool.")


if __name__ == "__main__":
    import json
    import anthropic

    client = anthropic.Anthropic()
    doc = "Paste or load your document text here."

    first_pass = extract_knowledge_graph(client, doc, domain="machine learning")
    additions = verify_knowledge_graph_extraction(
        client, doc, first_pass, domain="machine learning"
    )

    merged = {
        "entities": first_pass["entities"] + additions["entities"],
        "relations": first_pass["relations"] + additions["relations"],
        "research_gaps": first_pass["research_gaps"] + additions["research_gaps"],
    }
    print(json.dumps(merged, indent=2))
