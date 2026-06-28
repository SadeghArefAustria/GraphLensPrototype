"""
CLI: combine saved chunk_NNN_extraction.txt files into one deduplicated KG JSON.

Useful for reconstructing a final result from a chunked extraction run (PDF
--chunk-pages or text chunk_dir) without re-running the extraction — e.g. when
the run was interrupted before producing a merged JSON output.

Usage
-----
    python scripts/combine_chunks.py data/output/results_Alonso_chunks
    python scripts/combine_chunks.py data/output/results_Alonso_chunks --out data/output/results_Alonso.json
"""

import argparse
import json
import sys
from pathlib import Path

from graphlens.extractor import combine_chunk_extractions, pretty_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine chunk_NNN_extraction.txt files into one deduplicated KG JSON."
    )
    parser.add_argument("chunk_dir", help="Directory containing chunk_NNN_extraction.txt files.")
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write JSON results here (default: stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.is_dir():
        print(f"Error: directory not found: {chunk_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        result = combine_chunk_extractions(chunk_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

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
