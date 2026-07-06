"""
CLI: extract entities and relations from a plain-text file using Claude.

Splits the input text into chunks (see graphlens.extractor._chunk_text) and
extracts a knowledge graph from each chunk separately, then merges the results
— useful for text produced by scripts/pdf_to_text.py or any other plain-text
source too large to send to Claude in a single call.

Usage
-----
    python scripts/extract_kg_from_text.py <txt>
    python scripts/extract_kg_from_text.py <txt> --out data/output/results.json
    python scripts/extract_kg_from_text.py <txt> --title "Paper Title"
    python scripts/extract_kg_from_text.py <txt> --domain "academic research"
    python scripts/extract_kg_from_text.py <txt> --verify              # second pass, ~2x tokens
    python scripts/extract_kg_from_text.py <txt> --out data/output/results.json --save-text
    python scripts/extract_kg_from_text.py <txt> --chunk-size 8000 --save-chunks
    python scripts/extract_kg_from_text.py <txt> --chunk-size 0        # no chunking
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

from graphlens.extractor import extract_from_text, pretty_print, format_result
from graphlens.metadata import doc_id_for_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a knowledge graph from a plain-text file using Claude."
    )
    parser.add_argument("text_file", help="Path to the input .txt file.")
    parser.add_argument(
        "--title",
        metavar="TITLE",
        help="Optional document title included as context for the model.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write JSON results here (default: stdout).",
    )
    parser.add_argument(
        "--domain",
        metavar="DOMAIN",
        help='Optional domain hint, e.g. "academic research", "news" (default: none).',
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a second pass to catch missed entities/relations (~2x token usage).",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help=(
            "Also save the pretty-printed entities/relations to a .txt file "
            "alongside the JSON output (requires --out)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Split the text into ~N-character chunks and extract from each "
            "separately (default: graphlens.extractor.DEFAULT_CHUNK_SIZE). "
            "Pass 0 to disable chunking and extract from the whole text in "
            "one call."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Characters of trailing context carried from one chunk into the "
            "next (default: graphlens.extractor.DEFAULT_CHUNK_OVERLAP)."
        ),
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help=(
            "Save each chunk's text as chunk_NNN.txt and its extraction as "
            "chunk_NNN_extraction.txt under <stem>_chunks/."
        ),
    )
    parser.add_argument(
        "--doc-id-source",
        metavar="PATH",
        help=(
            "File to hash for the output's doc_id (default: the input .txt "
            "itself). Pass the original PDF here so the doc_id matches the "
            "one in that PDF's metadata record from scripts/pdf_metadata.py, "
            "letting the two be joined later."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text_path = Path(args.text_file)
    if not text_path.is_file():
        print(f"Error: file not found: {text_path}", file=sys.stderr)
        sys.exit(1)

    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"Error: file is empty: {text_path}", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size or None
    if args.chunk_overlap is not None:
        kwargs["chunk_overlap"] = args.chunk_overlap

    chunk_dir = None
    if args.save_chunks:
        base = Path(args.out) if args.out else text_path
        chunk_dir = base.parent / f"{base.stem}_chunks"

    client = anthropic.Anthropic()
    result = extract_from_text(
        client,
        text,
        title=args.title or "",
        domain=args.domain,
        verify=args.verify,
        chunk_dir=chunk_dir,
        **kwargs,
    )

    doc_id_source = Path(args.doc_id_source) if args.doc_id_source else text_path
    result = {"doc_id": doc_id_for_file(doc_id_source), **result}

    pretty_print(result)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_output, encoding="utf-8")
        print(f"Results written to {out_path}", file=sys.stderr)

        if args.save_text:
            txt_path = out_path.with_suffix(".txt")
            txt_path.write_text(format_result(result), encoding="utf-8")
            print(f"Text summary written to {txt_path}", file=sys.stderr)
    else:
        print(json_output)
        if args.save_text:
            print("Warning: --save-text requires --out; skipping.", file=sys.stderr)


if __name__ == "__main__":
    main()
