"""
Knowledge graph extraction from a PDF using the Claude API.

Usage:
    python extract_kg.py <path_to_pdf>
    python extract_kg.py <path_to_pdf> --out results.json
    python extract_kg.py <path_to_pdf> --file-id file_011ABC...  # reuse a previously uploaded PDF

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable must be set.
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Prompt
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

OUTPUT_SCHEMA = {
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
# Helpers
# ---------------------------------------------------------------------------


def upload_pdf(client: anthropic.Anthropic, pdf_path: Path) -> str:
    """Upload a PDF to the Files API and return the file_id."""
    print(f"Uploading {pdf_path.name} …", file=sys.stderr)
    with pdf_path.open("rb") as fh:
        result = client.beta.files.upload(
            file=(pdf_path.name, fh, "application/pdf"),
        )
    print(f"  → file_id: {result.id}", file=sys.stderr)
    return result.id


def extract(client: anthropic.Anthropic, file_id: str) -> dict:
    """Call Claude with the uploaded PDF and return parsed entities + relations."""
    print("Extracting entities and relations …", file=sys.stderr)

    # Stream the response so large PDFs don't hit HTTP timeouts.
    with client.beta.messages.stream(
        model="claude-opus-4-8",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        # Cache the stable system prompt – saves cost on repeated calls.
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
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
        # Force structured JSON output.
        output_config={
            "format": {
                "type": "json_schema",
                "schema": OUTPUT_SCHEMA,
            }
        },
        betas=["files-api-2025-04-14"],
    ) as stream:
        message = stream.get_final_message()

    # Report token usage.
    u = message.usage
    print(
        f"  Tokens — input: {u.input_tokens}, output: {u.output_tokens}, "
        f"cache_read: {getattr(u, 'cache_read_input_tokens', 0)}, "
        f"cache_write: {getattr(u, 'cache_creation_input_tokens', 0)}",
        file=sys.stderr,
    )

    # Parse the JSON from the first text block.
    text_block = next(b for b in message.content if b.type == "text")
    return json.loads(text_block.text)


def pretty_print(result: dict) -> None:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract entities and relations from a PDF using Claude."
    )
    parser.add_argument("pdf", nargs="?", help="Path to the input PDF file.")
    parser.add_argument(
        "--file-id",
        metavar="FILE_ID",
        help="Reuse a previously uploaded file (skip upload).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write JSON results to this file (default: stdout).",
    )
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Do not delete the uploaded file from the Files API after extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf and not args.file_id:
        print("Error: provide a PDF path or --file-id.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    file_id = args.file_id
    uploaded_now = False

    if not file_id:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            print(f"Error: file not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        file_id = upload_pdf(client, pdf_path)
        uploaded_now = True

    try:
        result = extract(client, file_id)
    finally:
        # Clean up the uploaded file unless the caller wants to keep it.
        if uploaded_now and not args.keep_file:
            try:
                client.beta.files.delete(file_id)
                print(f"Deleted file {file_id} from Files API.", file=sys.stderr)
            except Exception as exc:
                print(f"Warning: could not delete {file_id}: {exc}", file=sys.stderr)

    pretty_print(result)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(json_output, encoding="utf-8")
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()