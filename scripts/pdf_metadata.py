"""
CLI: build a metadata record for one or more PDF files.

Usage
-----
    python scripts/pdf_metadata.py paper.pdf
    python scripts/pdf_metadata.py paper.pdf --source "Acme Corp investor relations"
    python scripts/pdf_metadata.py paper.pdf --title "Q3 2026 Financial Report" \\
        --date-published 2026-06-15
    python scripts/pdf_metadata.py a.pdf b.pdf c.pdf --out-dir data/metadata/
    python scripts/pdf_metadata.py a.pdf b.pdf --out data/metadata/corpus.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from graphlens.metadata import build_pdf_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metadata record for one or more PDF files."
    )
    parser.add_argument("pdfs", nargs="+", metavar="PDF", help="Path(s) to input PDF file(s).")
    parser.add_argument(
        "--source",
        metavar="TEXT",
        help="Where the document came from, e.g. \"Acme Corp investor relations\". "
        "Applied to every PDF given.",
    )
    parser.add_argument(
        "--title",
        metavar="TEXT",
        help="Document title (only valid with a single PDF; overrides the PDF's own "
        "metadata title).",
    )
    parser.add_argument(
        "--date-published",
        metavar="YYYY-MM-DD",
        help="Publication date (only valid with a single PDF; overrides the PDF's "
        "own CreationDate metadata).",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        help="Write one <stem>.json file per PDF into this directory.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Append one JSON object per line to this .jsonl file (default: print "
        "to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.pdfs) > 1 and (args.title or args.date_published):
        print(
            "Error: --title and --date-published can only be used with a single PDF.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.out_dir and args.out:
        print("Error: use either --out-dir or --out, not both.", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    for pdf_arg in args.pdfs:
        pdf_path = Path(pdf_arg)
        if not pdf_path.is_file():
            print(f"Error: file not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        record = build_pdf_metadata(
            pdf_path,
            source=args.source,
            title=args.title,
            date_published=args.date_published,
        )
        records.append(record)

        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{pdf_path.stem}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {pdf_path.name}  ->  {out_path}", file=sys.stderr)

    if args.out_dir:
        return

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Appended {len(records)} record(s) to {out_path}", file=sys.stderr)
    else:
        for record in records:
            print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
