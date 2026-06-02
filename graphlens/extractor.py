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

    # From scraped text, with a verification second-pass
    result  = extract_from_text(client, text, title="...", domain="news", verify=True)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """You are a meticulous knowledge-graph extraction expert.

Your task is to extract a **complete and exhaustive** knowledge graph from the given document.

## Entities
Extract EVERY named entity in the document — do not skip anything, even if it seems minor.
For each entity provide:
- name        : canonical surface form (resolve pronouns and abbreviations to the full name)
- type        : one of PERSON, ORG, LOCATION, EVENT, CONCEPT, PRODUCT, OTHER
- description : one sentence grounded in the document

Entity extraction rules:
- Include every named person, organisation, place, product, technology, event, and concept.
- Resolve co-references: "the university", "it", "TU Graz" → always use the canonical name.
- Split compound references: "Magna and TU Graz" → two separate entities.

## Relations
Extract EVERY relation between entities — both explicit and clearly implied.
For each relation provide:
- subject   : entity name (must be in the entities list)
- predicate : short label in SCREAMING_SNAKE_CASE (e.g. WORKS_FOR, FUNDED_BY, PART_OF)
- object    : entity name (must be in the entities list)
- evidence  : verbatim quote or close paraphrase from the document

Relation extraction rules:
- Prefer specific predicates over generic ones (FOUNDED_BY rather than RELATED_TO).
- Extract transitive/implicit relations when they are clearly supported by the text.
- Do NOT fabricate relations — every relation must have an evidence quote.

## Critical
- It is far better to extract too much than to miss something important.
- Your output MUST be valid JSON that matches the schema exactly.{domain_section}"""

_DOMAIN_SECTION = """

## Domain focus
This document belongs to the domain: **{domain}**.
Pay special attention to entities and relations that are important in this domain."""

_VERIFICATION_SYSTEM_PROMPT = """You are a knowledge-graph quality reviewer.

You will be given:
1. The original document text.
2. A first-pass extraction (entities and relations already found).

Your job is to identify **only the entities and relations that were MISSED** in the first pass.
Do not repeat anything already in the first-pass result.
Apply the same exhaustive extraction rules as the original pass.

Return a JSON object with the same schema: {"entities": [...], "relations": [...]}
containing ONLY the additions — an empty list is fine if nothing was missed."""

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
    return _BASE_SYSTEM_PROMPT.format(domain_section=domain_section)


def _stream_extract(
    client:         anthropic.Anthropic,
    messages:       list[dict],
    system_prompt:  str,
    use_files_beta: bool = False,
) -> dict:
    """Run a single streaming extraction call and return the parsed JSON dict."""
    stream_kwargs = dict(
        model="claude-opus-4-8",
        max_tokens=8192,
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


def extract(
    client:  anthropic.Anthropic,
    file_id: str,
    *,
    domain: str | None = None,
    verify: bool = False,
) -> dict:
    """Run KG extraction on an already-uploaded PDF document.

    Parameters
    ----------
    client:  Anthropic API client.
    file_id: ID returned by :func:`upload_pdf`.
    domain:  Optional domain hint (e.g. ``"academic research"``, ``"news"``,
             ``"automotive engineering"``).  Helps the model focus on what
             matters in that domain.
    verify:  If ``True``, run a second Claude pass to find entities and
             relations missed in the first pass, then merge the results.
             Increases recall at the cost of roughly 2× token usage.
    """
    system_prompt = _build_system_prompt(domain)
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
    domain: str | None = None,
    verify: bool = False,
) -> dict:
    """Run KG extraction on a plain-text string (e.g. scraped web content).

    Parameters
    ----------
    client: Anthropic API client.
    text:   The document text to extract from.
    title:  Optional document title included as context for the model.
    domain: Optional domain hint (e.g. ``"news"``, ``"academic research"``).
    verify: If ``True``, run a second pass to find missed items and merge.
    """
    print(
        f"Extracting from text ({len(text):,} chars)"
        + (f" — {title!r}" if title else "")
        + (f"  [domain: {domain}]" if domain else "")
        + " (pass 1) …",
        file=sys.stderr,
    )

    system_prompt = _build_system_prompt(domain)
    header        = f"Document title: {title}\n\n" if title else ""
    user_message  = (
        f"{header}{text}\n\n"
        "---\n"
        "Extract ALL entities and relations from the text above. "
        "Be exhaustive — do not skip anything. "
        "Return strictly valid JSON matching the schema."
    )

    messages: list[dict] = [{"role": "user", "content": user_message}]
    result = _stream_extract(client, messages, system_prompt)

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
