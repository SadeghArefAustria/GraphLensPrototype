"""
Fetch recently published arXiv papers on a topic since a given date.

Example: all new papers in mobility/autonomous driving published since last Monday.
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
import feedparser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graphlens.metadata import build_pdf_metadata


def last_monday(reference=None):
    """Return the date of the most recent Monday (00:00 UTC)."""
    reference = reference or datetime.now(timezone.utc)
    days_since_monday = reference.weekday()  # Monday == 0
    monday = reference - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def build_query(keywords, categories, start_date, end_date=None):
    """
    keywords: list of phrases to search in title/abstract, OR'd together
    categories: list of arXiv category codes, OR'd together
    start_date / end_date: datetime objects (UTC)
    """
    end_date = end_date or datetime.now(timezone.utc)

    # Text search across title + abstract
    kw_query = " OR ".join(f'(ti:"{k}" OR abs:"{k}")' for k in keywords)

    # Category filter
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    # Date range filter (arXiv expects YYYYMMDDHHMM, UTC)
    date_query = (
        f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} "
        f"TO {end_date.strftime('%Y%m%d%H%M')}]"
    )

    full_query = f"({kw_query}) AND ({cat_query}) AND {date_query}"
    return full_query


def fetch_papers(query, max_results=50):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    feed = feedparser.parse(raw)
    return feed.entries


def get_pdf_url(entry):
    """Find the PDF link in an arXiv feed entry, falling back to id-based URL."""
    for link in entry.links:
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            return link.href
    # Fallback: derive from the entry id, e.g. http://arxiv.org/abs/2401.01234v1
    arxiv_id = entry.id.split("/abs/")[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def safe_filename(entry, max_len=100):
    """Build a filesystem-safe filename from the arXiv id and title."""
    arxiv_id = entry.id.split("/abs/")[-1]
    title = " ".join(entry.title.split())
    title = re.sub(r'[\\/*?:"<>|]', "", title)  # strip illegal filename chars
    title = title[:max_len].strip()
    return f"{arxiv_id}_{title}.pdf"


def _arxiv_date_published(entry):
    """Extract the arXiv submission date (YYYY-MM-DD) from a feed entry."""
    published = getattr(entry, "published", None)
    if not published:
        return None
    try:
        return datetime.strptime(published[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def save_metadata(entry, filepath):
    """Build and save a metadata record alongside the downloaded PDF at *filepath*."""
    arxiv_id = entry.id.split("/abs/")[-1]
    title = " ".join(entry.title.split())

    record = build_pdf_metadata(
        filepath,
        source="arXiv",
        title=title,
        date_published=_arxiv_date_published(entry),
    )
    record["arxiv_id"] = arxiv_id
    record["authors"] = [a.name for a in entry.authors]
    record["abstract_url"] = entry.link

    meta_path = Path(filepath).with_suffix(".json")
    meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def download_papers(entries, out_dir="arxiv_papers", delay=3):
    """
    Download the PDF for each entry into out_dir, along with a metadata
    record (title, authors, published date, sha256, etc.) as a sibling
    .json file.
    `delay` seconds between requests, per arXiv's usage guidelines.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, entry in enumerate(entries, 1):
        pdf_url = get_pdf_url(entry)
        filename = safe_filename(entry)
        filepath = os.path.join(out_dir, filename)

        if os.path.exists(filepath):
            print(f"[{i}/{len(entries)}] Skipping (already exists): {filename}")
            continue

        print(f"[{i}/{len(entries)}] Downloading: {filename}")
        try:
            req = urllib.request.Request(
                pdf_url, headers={"User-Agent": "Mozilla/5.0 (arxiv-fetch-script)"}
            )
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
            meta_path = save_metadata(entry, filepath)
            print(f"  Metadata saved: {os.path.basename(meta_path)}")
        except Exception as e:
            print(f"  Failed: {e}")

        time.sleep(delay)  # be polite to arXiv's servers

    print(f"\nDone. PDFs saved to: {os.path.abspath(out_dir)}")


def print_results(entries):
    print(f"Found {len(entries)} papers\n")
    for e in entries:
        title = " ".join(e.title.split())
        authors = ", ".join(a.name for a in e.authors)
        published = e.published
        link = e.link
        print(f"- {title}")
        print(f"  Authors: {authors}")
        print(f"  Published: {published}")
        print(f"  Link: {link}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch recently published arXiv papers on a topic since a given date."
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["driving safety", "autonomous vehicles", "self-driving cars"],
        help="Phrases to search in title/abstract (space-separated; quote multi-word "
        "phrases), OR'd together.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Only include papers submitted on or after this date, YYYY-MM-DD "
        "(default: most recent Monday).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    keywords = args.keywords
    categories = ["cs.RO", "cs.CV", "cs.LG", "eess.SY"]
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_date = last_monday()

    query = build_query(keywords, categories, start_date)
    print(f"Query: {query}\n")

    entries = fetch_papers(query, max_results=50)
    print_results(entries)

    # ---- Download PDFs into a new folder ----
    download_papers(entries, out_dir="arxiv_papers", delay=3)