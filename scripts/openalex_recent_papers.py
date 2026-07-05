"""
Fetch recently published papers from OpenAlex (via pyalex) and download
whichever ones have full-text PDF content available.

Example: all new papers in mobility/autonomous driving published since last Monday.

Requires a free OpenAlex API key (mandatory since Feb 13, 2026):
  https://openalex.org/settings/api
"""

import os
import re
from datetime import datetime, timedelta, timezone

import pyalex
from pyalex import Works

# ---- Set your API key ----
pyalex.config.api_key = "6gIDj0PpxIHKe5kluOYnII"  # <-- replace this
# Optional but polite: identify yourself for the "polite pool" (faster, more reliable)
pyalex.config.email = "sadegharef@gmail.com"  # <-- replace this


def last_monday(reference=None):
    """Return the date of the most recent Monday."""
    reference = reference or datetime.now(timezone.utc)
    days_since_monday = reference.weekday()  # Monday == 0
    monday = reference - timedelta(days=days_since_monday)
    return monday.date()


def search_papers(query, start_date, end_date=None, per_page=200):
    """
    query: search string, e.g. 'autonomous driving'
    start_date / end_date: date objects
    """
    end_date = end_date or datetime.now(timezone.utc).date()

    pager = (
        Works()
        .search(query)
        .filter(from_publication_date=str(start_date), to_publication_date=str(end_date))
        .sort(publication_date="desc")
        .paginate(per_page=per_page, n_max=None)  # None = fetch all matching pages
    )

    all_works = []
    for page in pager:
        all_works.extend(page)
        print(f"Retrieved {len(all_works)} papers so far...")

    return all_works


def print_results(works):
    print(f"\nFound {len(works)} papers\n")
    for w in works:
        title = w.get("display_name", "Untitled")
        date = w.get("publication_date", "unknown date")
        authors = ", ".join(
            a["author"]["display_name"] for a in w.get("authorships", [])
        )
        oa = w.get("open_access", {})
        print(f"- {title}")
        print(f"  Published: {date} | Authors: {authors}")
        print(f"  Open access: {oa.get('is_oa')} | {w.get('id')}\n")


def safe_filename(work, max_len=100):
    title = re.sub(r'[\\/*?:"<>|]', "", work.get("display_name", "untitled"))
    title = title[:max_len].strip()
    work_id = work.get("id", "").replace("https://openalex.org/", "")
    return f"{work_id}_{title}.pdf"


def download_papers(works, out_dir="openalex_papers"):
    """Download PDF content for each work, where OpenAlex has it available."""
    os.makedirs(out_dir, exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0
    for i, work in enumerate(works, 1):
        filename = safe_filename(work)
        filepath = os.path.join(out_dir, filename)

        if os.path.exists(filepath):
            print(f"[{i}/{len(works)}] Skipping (already exists): {filename}")
            skipped += 1
            continue

        try:
            # pyalex fetches the full Work object's .pdf accessor and downloads it
            w = Works()[work["id"]]
            w.pdf.download(filepath)
            print(f"[{i}/{len(works)}] Downloaded: {filename}")
            downloaded += 1
        except Exception as e:
            print(f"[{i}/{len(works)}] No PDF available or failed: {e}")
            failed += 1

    print(
        f"\nDone. {downloaded} downloaded, {skipped} skipped (already existed), "
        f"{failed} unavailable/failed.\nSaved to: {os.path.abspath(out_dir)}"
    )


if __name__ == "__main__":
    # ---- Configure your search here ----
    query = "driving safety AND autonomous vehicles AND self-driving cars"
    start_date = "2026-06-20"
    # -------------------------------------

    print(f"Searching for papers since {start_date}...\n")
    works = search_papers(query, start_date)

    print_results(works)

    # ---- Download available PDFs into a new folder ----
    download_papers(works, out_dir="openalex_papers")
