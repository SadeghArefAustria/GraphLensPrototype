"""
Build a metadata record for a source PDF document.

Typical usage
-------------
    from pathlib import Path
    from graphlens.metadata import build_pdf_metadata

    meta = build_pdf_metadata(
        Path("Q3_report_acme_corp.pdf"),
        source="Acme Corp investor relations",
    )
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path

from pypdf import PdfReader

DOC_ID_LENGTH = 12  # hex chars of the sha256 digest used as the short doc_id


def sha256_of_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of *path*, reading it in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_from_sha256(sha256: str) -> str:
    """Derive the short ``doc_id`` used across the pipeline from a sha256 digest."""
    return sha256[:DOC_ID_LENGTH]


def doc_id_for_file(path: str | Path) -> str:
    """Derive the short ``doc_id`` for *path* from its content hash.

    Use this to stamp the same ``doc_id`` used in a file's metadata record
    (see :func:`build_pdf_metadata`) onto other artifacts derived from it
    (e.g. an extracted knowledge graph), so they can be joined later.
    """
    return doc_id_from_sha256(sha256_of_file(path))


def _parse_pdf_date(raw: str | None) -> str | None:
    """Parse a PDF metadata date string (``D:YYYYMMDD...``) into ``YYYY-MM-DD``."""
    if not raw:
        return None
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def build_pdf_metadata(
    pdf_path: str | Path,
    *,
    source: str | None = None,
    title: str | None = None,
    date_published: str | None = None,
) -> dict:
    """Build a metadata record for *pdf_path*.

    Parameters
    ----------
    pdf_path:       Path to the local PDF file.
    source:         Where the document came from (e.g. "Acme Corp investor
                     relations"). Not derivable from the file — pass it in.
    title:          Document title. Falls back to the PDF's own metadata
                     title, then to the filename stem, if not given.
    date_published: Publication date (``YYYY-MM-DD``). Falls back to the
                     PDF's ``CreationDate`` metadata field if not given.

    Returns
    -------
    A dict with the fields: ``doc_id``, ``original_filename``, ``title``,
    ``source``, ``date_published``, ``date_ingested``, ``page_count``,
    ``sha256``.
    """
    pdf_path = Path(pdf_path)
    sha256 = sha256_of_file(pdf_path)

    reader = PdfReader(str(pdf_path))
    pdf_meta = reader.metadata or {}

    return {
        "doc_id": doc_id_from_sha256(sha256),
        "original_filename": pdf_path.name,
        "title": title or pdf_meta.title or pdf_path.stem,
        "source": source or "",
        "date_published": date_published or _parse_pdf_date(pdf_meta.get("/CreationDate")) or "",
        "date_ingested": datetime.now().date().isoformat(),
        "page_count": len(reader.pages),
        "sha256": sha256,
    }
