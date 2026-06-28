"""
CLI: extract entities and relations from a PDF using Claude.

Usage
-----
    python scripts/extract_kg.py <pdf>
    python scripts/extract_kg.py <pdf> --out data/output/results.json
    python scripts/extract_kg.py --file-id file_011ABC...        # reuse upload
    python scripts/extract_kg.py <pdf> --keep-file               # don't delete from Files API
    python scripts/extract_kg.py <pdf> --domain "academic research"
    python scripts/extract_kg.py <pdf> --verify                  # second pass, ~2x tokens
    python scripts/extract_kg.py <pdf> --out data/output/results.json --save-text
    python scripts/extract_kg.py <pdf> --chunk-pages 5 --save-chunks  # per-chunk extraction
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

from graphlens.extractor import upload_pdf, extract, pretty_print, format_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a knowledge graph from a PDF using Claude."
    )
    parser.add_argument("pdf", nargs="?", help="Path to the input PDF file.")
    parser.add_argument(
        "--file-id",
        metavar="FILE_ID",
        help="Reuse a previously uploaded file (skips upload).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write JSON results here (default: stdout).",
    )
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Do not delete the uploaded file from the Files API after extraction.",
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
        "--chunk-pages",
        type=int,
        metavar="N",
        help=(
            "Split the PDF into chunks of N pages each and extract from each "
            "separately for finer-grained recall (requires a PDF path, not --file-id)."
        ),
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help=(
            "With --chunk-pages, save each page-range chunk as chunk_NNN.pdf and "
            "its extraction as chunk_NNN_extraction.txt under <stem>_chunks/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf and not args.file_id:
        print("Error: provide a PDF path or --file-id.", file=sys.stderr)
        sys.exit(1)

    if args.chunk_pages and not args.pdf:
        print(
            "Error: --chunk-pages requires a PDF path (not --file-id alone).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.pdf and not Path(args.pdf).is_file():
        print(f"Error: file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    file_id         = args.file_id
    uploaded_now    = False
    needs_whole_pdf = not args.chunk_pages or args.verify

    if not file_id and needs_whole_pdf:
        file_id      = upload_pdf(client, Path(args.pdf))
        uploaded_now = True

    chunk_dir = None
    if args.save_chunks and args.chunk_pages:
        base      = Path(args.out) if args.out else Path(args.pdf)
        chunk_dir = base.parent / f"{base.stem}_chunks"

    try:
        result = extract(
            client, file_id,
            domain=args.domain,
            verify=args.verify,
            pdf_path=args.pdf,
            chunk_pages=args.chunk_pages,
            chunk_dir=chunk_dir,
        )
    finally:
        if uploaded_now and not args.keep_file:
            try:
                client.beta.files.delete(file_id)
                print(f"Deleted remote file {file_id}.", file=sys.stderr)
            except Exception as exc:
                print(f"Warning: could not delete {file_id}: {exc}", file=sys.stderr)

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