"""
CLI: scrape one or more web pages and extract a knowledge graph from each.

Usage
-----
    # Single URL → stdout
    python scripts/scrape_kg.py https://example.com/article

    # Multiple URLs, merge into one KG, save to file
    python scripts/scrape_kg.py https://a.com https://b.com \\
        --merge --out data/output/merged.json

    # Read URLs from a text file (one per line)
    python scripts/scrape_kg.py --url-file urls.txt --out data/output/results.json

    # Don't check robots.txt (use responsibly)
    python scripts/scrape_kg.py https://example.com --no-robots

    # Save one JSON per URL (auto-named from the URL slug)
    python scripts/scrape_kg.py https://a.com https://b.com --out-dir data/output/

    # Also save the scraped plain text alongside each JSON
    python scripts/scrape_kg.py https://a.com --out data/output/results.json --save-text

    # Increase delay between requests
    python scripts/scrape_kg.py https://a.com https://b.com --delay 2.5
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import anthropic

from graphlens.scraper import scrape_many, ScrapedPage
from graphlens.extractor import extract_from_text, pretty_print
from graphlens.ml.graph_builder import KGGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_to_filename(url: str) -> str:
    """Derive a safe filename stem from a URL."""
    parsed   = urlparse(url)
    slug     = (parsed.netloc + parsed.path).strip("/").replace("/", "_")
    slug     = "".join(c if c.isalnum() or c in "-_." else "_" for c in slug)
    return slug[:80] or "page"


def _save_text(page: "ScrapedPage", json_path: Path) -> None:
    """Write scraped text to a .txt file next to *json_path*."""
    txt_path = json_path.with_suffix(".txt")
    header   = f"URL: {page.url}\nTitle: {page.title}\n{'=' * 60}\n\n"
    txt_path.write_text(header + page.text, encoding="utf-8")
    print(f"  Text saved to {txt_path}", file=sys.stderr)


def _merge_kg_dicts(dicts: list[dict]) -> dict:
    """Merge a list of KG dicts using KGGraph.merge to deduplicate."""
    graphs = [KGGraph(d) for d in dicts]
    merged = KGGraph.merge(graphs)
    return {
        "entities": [
            {
                "name":        name,
                "type":        merged._entity_meta.get(name, {}).get("type", "OTHER"),
                "description": merged._entity_meta.get(name, {}).get("description", ""),
            }
            for name in merged.id_to_entity
        ],
        "relations": [
            {
                "subject":   t.head,
                "predicate": t.relation,
                "object":    t.tail,
                "evidence":  "",
            }
            for t in merged.triples
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape web pages and extract knowledge graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument(
        "urls", nargs="*", metavar="URL",
        help="One or more URLs to scrape.",
    )
    url_group.add_argument(
        "--url-file", metavar="FILE",
        help="Text file with one URL per line.",
    )

    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all extracted KGs into a single output.",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Output JSON path (used with --merge or a single URL).",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR",
        help="Output directory for per-URL JSON files (ignored when --merge).",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between HTTP requests (default: 1.0).",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Read timeout per request in seconds (default: 60).",
    )
    parser.add_argument(
        "--connect-timeout", type=int, default=15,
        help="TCP connect timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--no-robots", action="store_true",
        help="Ignore robots.txt (use responsibly and only on sites you control).",
    )
    parser.add_argument(
        "--max-chars", type=int, default=120_000,
        help="Truncate page text beyond this many characters (default: 120000).",
    )
    parser.add_argument(
        "--save-text", action="store_true",
        help=(
            "Save the scraped plain text to a .txt file alongside each JSON output. "
            "The text file gets the same name as the JSON but with a .txt extension."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Collect URLs
    if args.url_file:
        url_file = Path(args.url_file)
        if not url_file.is_file():
            print(f"Error: URL file not found: {url_file}", file=sys.stderr)
            sys.exit(1)
        urls = [u.strip() for u in url_file.read_text().splitlines() if u.strip()]
    else:
        urls = args.urls

    if not urls:
        print("Error: no URLs provided.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Scrape
    # ------------------------------------------------------------------
    pages = scrape_many(
        urls,
        delay=args.delay,
        timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        check_robots=not args.no_robots,
        max_chars=args.max_chars,
    )

    successful = [p for p in pages if p.ok]
    if not successful:
        print("Error: all URLs failed to scrape.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nScraping done: {len(successful)}/{len(pages)} pages fetched.\n",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Extract KG from each page
    # ------------------------------------------------------------------
    client = anthropic.Anthropic()
    kg_results: list[tuple[ScrapedPage, dict]] = []

    for page in successful:
        result = extract_from_text(client, page.text, title=page.title)
        pretty_print(result)
        kg_results.append((page, result))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.merge or len(kg_results) == 1:
        merged   = _merge_kg_dicts([r for _, r in kg_results])
        json_out = json.dumps(merged, indent=2, ensure_ascii=False)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_out, encoding="utf-8")
            print(f"\nMerged KG written to {out_path}", file=sys.stderr)

            if args.save_text:
                if len(kg_results) == 1:
                    # Single page — save its text next to the JSON
                    _save_text(kg_results[0][0], out_path)
                else:
                    # Multiple pages merged — save each text individually
                    for page, _ in kg_results:
                        stem     = _url_to_filename(page.url)
                        txt_path = out_path.parent / f"{stem}.txt"
                        header   = f"URL: {page.url}\nTitle: {page.title}\n{'=' * 60}\n\n"
                        txt_path.write_text(header + page.text, encoding="utf-8")
                        print(f"  Text saved to {txt_path}", file=sys.stderr)
        else:
            print(json_out)

    else:
        # One file per URL
        out_dir = Path(args.out_dir) if args.out_dir else Path("data/output")
        out_dir.mkdir(parents=True, exist_ok=True)

        for page, result in kg_results:
            stem     = _url_to_filename(page.url)
            out_path = out_dir / f"{stem}.json"
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  {page.url}  →  {out_path}", file=sys.stderr)

            if args.save_text:
                _save_text(page, out_path)

        print(f"\n{len(kg_results)} files written to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
