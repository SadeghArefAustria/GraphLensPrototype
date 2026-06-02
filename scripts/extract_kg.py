"""
CLI: extract entities and relations from a PDF using Claude.

Usage
-----
    python scripts/extract_kg.py <pdf>
    python scripts/extract_kg.py <pdf> --out data/output/results.json
    python scripts/extract_kg.py --file-id file_011ABC...        # reuse upload
    python scripts/extract_kg.py <pdf> --keep-file               # don't delete from Files API
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

from graphlens.extractor import upload_pdf, extract, pretty_print


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf and not args.file_id:
        print("Error: provide a PDF path or --file-id.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    file_id      = args.file_id
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
    else:
        print(json_output)


if __name__ == "__main__":
    main()