"""
CLI: extract the text content of a PDF and save it to a .txt file.

Usage
-----
    python scripts/pdf_to_text.py <pdf>
    python scripts/pdf_to_text.py <pdf> --out data/output/paper.txt
"""

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and save it to a text file."
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Path to write the extracted text (default: <pdf>.txt next to the input file).",
    )
    return parser.parse_args()


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = extract_text(pdf_path)
    out_path.write_text(text, encoding="utf-8")

    print(f"Saved extracted text to {out_path}")


if __name__ == "__main__":
    main()
