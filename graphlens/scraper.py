"""
Web scraper for GraphLens.

Fetches one or more URLs, strips boilerplate HTML (nav, footer, scripts, ads),
and returns clean article text ready for knowledge-graph extraction.

Typical usage
-------------
    from graphlens.scraper import scrape, scrape_many

    page  = scrape("https://example.com/article")
    pages = scrape_many(["https://a.com", "https://b.com"], delay=1.5)

    for p in pages:
        if p.ok:
            result = extract_from_text(client, p.text, title=p.title)
"""

from __future__ import annotations

import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise ImportError(
        "beautifulsoup4 is not installed. Run: pip install graphlens[scrape]"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Headers that make the request look like a real browser visit.
# Many sites block or slow-respond to requests that lack these.
_BROWSER_HEADERS = {
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT":                       "1",
}

# Tags that carry boilerplate rather than article content
_BOILERPLATE_TAGS = frozenset({
    "script", "style", "nav", "header", "footer", "aside",
    "form", "button", "iframe", "noscript",
})

# Candidate content containers tried in priority order
_CONTENT_SELECTORS = ["main", "article", "[role=main]", "div.content",
                      "div.article", "div.post", "body"]

# Hard cap on extracted text length to avoid runaway token costs
MAX_TEXT_CHARS = 120_000


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class ScrapedPage:
    """Result of scraping a single URL."""
    url:         str
    title:       str
    text:        str
    status_code: int = 200
    error:       str = ""
    truncated:   bool = False

    @property
    def ok(self) -> bool:
        """True when the page was fetched and parsed without errors."""
        return not self.error and self.status_code < 400


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_html(html: str, base_url: str) -> tuple[str, str]:
    """Return *(title, cleaned_text)* from raw HTML."""
    # Prefer lxml for speed; fall back to the built-in html.parser
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else base_url

    # Remove boilerplate elements in-place
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    # Find the best content container
    container = None
    for selector in _CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container:
            break
    container = container or soup

    # Collect non-empty text nodes
    chunks: list[str] = []
    for element in container.find_all(string=True):
        chunk = element.get_text(" ", strip=True)
        if chunk:
            chunks.append(chunk)

    text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    return title, text


def _robots_allowed(url: str, user_agent: str) -> bool:
    """Return True when robots.txt permits fetching *url*."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # robots.txt unreachable → assume allowed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape(
    url: str,
    *,
    timeout:        int  = 60,
    connect_timeout: int = 15,
    user_agent:     str  = DEFAULT_USER_AGENT,
    check_robots:   bool = True,
    max_chars:      int  = MAX_TEXT_CHARS,
) -> ScrapedPage:
    """Fetch *url* and return a :class:`ScrapedPage` with clean text.

    Parameters
    ----------
    url:             The page to fetch.
    timeout:         Read timeout in seconds (default: 60).
    connect_timeout: TCP connect timeout in seconds (default: 15).
    user_agent:      User-agent string sent in the request.
    check_robots:    Whether to honour robots.txt directives.
    max_chars:       Truncate extracted text beyond this length.
    """
    if check_robots and not _robots_allowed(url, user_agent):
        return ScrapedPage(
            url=url, title="", text="",
            status_code=403,
            error=f"Blocked by robots.txt: {url}",
        )

    headers = {**_BROWSER_HEADERS, "User-Agent": user_agent}
    http_timeout = httpx.Timeout(timeout, connect=connect_timeout)

    try:
        with httpx.Client(
            headers=headers,
            timeout=http_timeout,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ScrapedPage(
            url=url, title="", text="",
            status_code=exc.response.status_code,
            error=str(exc),
        )
    except Exception as exc:
        return ScrapedPage(url=url, title="", text="", error=str(exc))

    content_type = resp.headers.get("content-type", "")

    if "html" in content_type:
        title, text = _clean_html(resp.text, url)
    else:
        # Plain text, JSON, Markdown, etc. — use as-is
        title = url
        text  = resp.text.strip()

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return ScrapedPage(
        url=url,
        title=title,
        text=text,
        status_code=resp.status_code,
        truncated=truncated,
    )


def scrape_many(
    urls: list[str],
    *,
    delay:           float = 1.0,
    timeout:         int   = 60,
    connect_timeout: int   = 15,
    user_agent:      str   = DEFAULT_USER_AGENT,
    check_robots:    bool  = True,
    max_chars:       int   = MAX_TEXT_CHARS,
) -> list[ScrapedPage]:
    """Fetch multiple *urls* sequentially with a polite delay between requests.

    Parameters
    ----------
    urls:            List of URLs to fetch.
    delay:           Seconds to wait between requests (be polite to servers).
    timeout:         Read timeout per request in seconds (default: 60).
    connect_timeout: TCP connect timeout in seconds (default: 15).
    user_agent:      User-agent string.
    check_robots:    Whether to honour robots.txt.
    max_chars:       Per-page text truncation limit.
    """
    results: list[ScrapedPage] = []

    for i, url in enumerate(urls):
        print(f"Scraping [{i + 1}/{len(urls)}] {url} …", file=sys.stderr)
        page = scrape(
            url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            user_agent=user_agent,
            check_robots=check_robots,
            max_chars=max_chars,
        )
        if page.ok:
            trunc = " (truncated)" if page.truncated else ""
            print(
                f"  ✓ {len(page.text):,} chars{trunc}  title: {page.title!r}",
                file=sys.stderr,
            )
        else:
            print(f"  ✗ {page.error}", file=sys.stderr)

        results.append(page)

        if i < len(urls) - 1:
            time.sleep(delay)

    return results
