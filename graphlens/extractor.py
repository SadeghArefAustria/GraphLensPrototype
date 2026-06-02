"""
Claude-based knowledge-graph extraction from documents.

Typical usage
-------------
    import anthropic
    from graphlens.extractor import upload_pdf, extract

    client = anthropic.Anthropic()
    file_id = upload_pdf(client, Path("paper.pdf"))
    result  = extract(client, file_id)
    # result = {"entities": [...], "relations": [...]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Prompt & output schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledge-graph extraction expert.

Given a document, extract:
1. **Entities** – named things (people, organisations, locations, products,
   concepts, events, etc.).  For each entity provide:
   - name        : canonical surface form
   - type        : entity type (PERSON, ORG, LOCATION, EVENT, CONCEPT, PRODUCT, OTHER)
   - description : one-sentence description grounded in the document

2. **Relations** – directed relationships between entities.  For each relation:
   - subject     : entity name (must appear in the entities list)
   - predicate   : short relation label in SCREAMING_SNAKE_CASE
                   (e.g. WORKS_FOR, LOCATED_IN, PART_OF, FOUNDED_BY)
   - object      : entity name (must appear in the entities list)
   - evidence    : a short verbatim quote or paraphrase from the document

Rules:
- Only extract information that is explicitly stated or clearly implied.
- Be thorough but precise; prefer quality over quantity.
- Use consistent entity names (resolve co-references to the canonical form).
- Your response MUST be valid JSON conforming to the schema provided."""

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
# Public API
# ---------------------------------------------------------------------------


def upload_pdf(client: anthropic.Anthropic, pdf_path: Path) -> str:
    """Upload *pdf_path* to the Files API and return the remote ``file_id``.

    The caller is responsible for deleting the file when no longer needed::

        client.beta.files.delete(file_id)
    """
    print(f"Uploading {pdf_path.name} …", file=sys.stderr)
    with pdf_path.open("rb") as fh:
        result = client.beta.files.upload(
            file=(pdf_path.name, fh, "application/pdf"),
        )
    print(f"  → file_id: {result.id}", file=sys.stderr)
    return result.id


def extract(client: anthropic.Anthropic, file_id: str) -> dict:
    """Run KG extraction on an already-uploaded document.

    Returns a dict with keys ``entities`` and ``relations``.
    """
    print("Extracting entities and relations …", file=sys.stderr)

    with client.beta.messages.stream(
        model="claude-opus-4-8",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # save cost on repeated calls
            }
        ],
        messages=[
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
                            "Extract all entities and relations from the document above. "
                            "Return strictly valid JSON that matches the schema."
                        ),
                    },
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": OUTPUT_SCHEMA,
            }
        },
        betas=["files-api-2025-04-14"],
    ) as stream:
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


def pretty_print(result: dict) -> None:
    """Print a human-readable summary of extracted entities and relations."""
    entities  = result.get("entities", [])
    relations = result.get("relations", [])

    print(f"\n{'=' * 60}")
    print(f"ENTITIES  ({len(entities)})")
    print("=" * 60)
    for e in entities:
        print(f"  [{e['type']}] {e['name']}")
        print(f"    {e['description']}")

    print(f"\n{'=' * 60}")
    print(f"RELATIONS ({len(relations)})")
    print("=" * 60)
    for r in relations:
        print(f"  {r['subject']}  —[{r['predicate']}]→  {r['object']}")
        print(f"    evidence: {r['evidence']}")
    print()