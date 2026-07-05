"""
Fetch recently published papers from the Semantic Scholar Academic Graph API
and download whichever ones have an open-access PDF available.

Example: all new papers in mobility/autonomous driving published since last Monday.
"""

import os
import re
import time
import json
import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Optional: get a free API key at https://www.semanticscholar.org/product/api#api-key
# Without one you're rate-limited much more aggressively (shared pool).
API_KEY = None  # e.g. "your-key-here"


def last_monday(reference=None):
    """Return the date of the most recent Monday."""
    reference = reference or datetime.now(timezone.utc)
    days_since_monday = reference.weekday()  # Monday == 0
    monday = reference - timedelta(days=days_since_monday)
    return monday.date()


def search_papers(query, start_date, end_date=None, fields=None, sort="publicationDate:desc"):
    """
    query: search string, e.g. '"autonomous driving" | "mobility"'
    start_date / end_date: date objects
    fields: comma-separated field list
    """
    end_date = end_date or datetime.now(timezone.utc).date()
    fields = fields or "title,url,authors,publicationDate,openAccessPdf,abstract"

    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {
        "query": query,
        "fields": fields,
        "publicationDateOrYear": f"{start_date}:{end_date}",
        "sort": sort,
    }

    all_papers = []
    while True:
        resp = requests.get(BASE_URL, params=params, headers=headers)
        if resp.status_code != 200:
            print(f"Request failed ({resp.status_code}): {resp.text}")
            break

        data = resp.json()
        batch = data.get("data", [])
        all_papers.extend(batch)
        print(f"Retrieved {len(all_papers)} / ~{data.get('total', '?')} papers so far...")

        token = data.get("token")
        if not token:
            break
        params["token"] = token
        time.sleep(1)  # be polite / respect rate limits

    return all_papers


def print_results(papers):
    print(f"\nFound {len(papers)} papers\n")
    for p in papers:
        title = p.get("title", "Untitled")
        date = p.get("publicationDate", "unknown date")
        authors = ", ".join(a["name"] for a in p.get("authors", []))
        has_pdf = "yes" if p.get("openAccessPdf") else "no"
        print(f"- {title}")
        print(f"  Published: {date} | Authors: {authors} | PDF available: {has_pdf}")
        print(f"  {p.get('url')}\n")


def safe_filename(paper, max_len=100):
    title = re.sub(r'[\\/*?:"<>|]', "", paper.get("title", "untitled"))
    title = title[:max_len].strip()
    paper_id = paper.get("paperId", "unknown")
    return f"{paper_id}_{title}.pdf"


def download_papers(papers, out_dir="semanticscholar_papers", delay=1):
    """Download PDFs for papers that have an openAccessPdf link."""
    os.makedirs(out_dir, exist_ok=True)

    downloadable = [p for p in papers if p.get("openAccessPdf") and p["openAccessPdf"].get("url")]
    print(f"\n{len(downloadable)} of {len(papers)} papers have an open-access PDF available.\n")

    for i, paper in enumerate(downloadable, 1):
        pdf_url = paper["openAccessPdf"]["url"]
        filename = safe_filename(paper)
        filepath = os.path.join(out_dir, filename)

        if os.path.exists(filepath):
            print(f"[{i}/{len(downloadable)}] Skipping (already exists): {filename}")
            continue

        print(f"[{i}/{len(downloadable)}] Downloading: {filename}")
        try:
            resp = requests.get(
                pdf_url,
                headers={"User-Agent": "Mozilla/5.0 (semantic-scholar-fetch-script)"},
                timeout=30,
            )
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"  Failed: {e}")

        time.sleep(delay)

    print(f"\nDone. PDFs saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    # ---- Configure your search here ----
    # '|' = OR, '+' = AND, quotes = exact phrase (see Semantic Scholar query syntax)
    query = '"autonomous driving" | "self-driving" | "mobility"'
    start_date = last_monday()
    # -------------------------------------

    print(f"Searching for papers since {start_date}...\n")
    papers = search_papers(query, start_date)

    print_results(papers)

    # ---- Download available open-access PDFs into a new folder ----
    download_papers(papers, out_dir="semanticscholar_papers", delay=1)
