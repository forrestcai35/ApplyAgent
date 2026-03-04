"""Job deduplication: identifies and marks duplicate postings across sources.

The same position often appears on multiple job boards (Indeed, LinkedIn,
Glassdoor, etc.) with different URLs. This module detects those near-duplicates
using a fingerprint of normalised company + title, and marks all but the best
version with ``duplicate_of = <canonical_url>``.

Downstream stages (enrich, score, tailor, cover, apply) skip rows where
``duplicate_of IS NOT NULL``, so the same job is only processed once.

Fingerprinting strategy:
  1. Normalise company name (lowercase, strip legal suffixes like Inc/LLC/Ltd)
  2. Normalise title (lowercase, collapse whitespace)
  3. SHA-256 hash of ``company|title``
  4. Group by fingerprint — pick the "best" row as canonical, mark the rest

Best-row selection prefers (in order):
  - Already enriched (has full_description)
  - Has an application_url
  - Has a longer description
  - Earlier discovered_at timestamp
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections import defaultdict

from applyagent.database import get_connection

log = logging.getLogger(__name__)

# Legal suffixes stripped during company normalisation
_LEGAL_SUFFIXES = re.compile(
    r",?\s*\b(inc\.?|incorporated|llc|ltd\.?|limited|corp\.?|corporation|"
    r"co\.?|company|group|holdings|plc|gmbh|ag|sa|s\.?a\.?|"
    r"l\.?p\.?|llp|pllc)\b\.?",
    re.IGNORECASE,
)

# Noise tokens to remove from titles
_TITLE_NOISE = re.compile(
    r"\b(remote|hybrid|onsite|on-site|full-time|full time|part-time|part time|"
    r"contract|temp|temporary|permanent|ft|pt|fte)\b",
    re.IGNORECASE,
)

# Multiple whitespace / special chars
_WHITESPACE = re.compile(r"[\s\-_/\\]+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalise_company(raw: str | None) -> str:
    """Collapse a company name to a canonical lowercase form."""
    if not raw:
        return ""
    s = raw.lower().strip()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def normalise_title(raw: str | None) -> str:
    """Collapse a job title to a canonical lowercase form."""
    if not raw:
        return ""
    s = raw.lower().strip()
    # Strip "Company — " prefix used by github_repos
    if " — " in s:
        s = s.split(" — ", 1)[1]
    s = _TITLE_NOISE.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def fingerprint(company: str | None, title: str | None) -> str | None:
    """Generate a dedup fingerprint from company + title.

    Returns None if there isn't enough information to fingerprint
    (e.g. both are empty/None).
    """
    nc = normalise_company(company)
    nt = normalise_title(title)

    if not nt:
        return None
    if not nc:
        # Without a company name we can still fingerprint on title alone,
        # but only if the title is specific enough (>3 words) to avoid
        # collisions like "Software Engineer" at different companies.
        if len(nt.split()) <= 3:
            return None

    key = f"{nc}|{nt}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _pick_best(rows: list[sqlite3.Row]) -> sqlite3.Row:
    """From a group of duplicate rows, pick the best canonical version."""
    def _score(row: sqlite3.Row) -> tuple:
        has_full = 1 if row["full_description"] else 0
        has_apply = 1 if row["application_url"] else 0
        desc_len = len(row["description"] or "")
        # Earlier discovery is better (sort ascending by inverting)
        disc = row["discovered_at"] or "9999"
        return (has_full, has_apply, desc_len, -len(disc), disc)

    return max(rows, key=_score)


def run_dedup(conn: sqlite3.Connection | None = None) -> dict:
    """Scan the jobs table and mark near-duplicates.

    Skips rows that are already marked as duplicates or have been applied to.

    Returns:
        Dict with stats: scanned, groups, marked, already_marked.
    """
    if conn is None:
        conn = get_connection()

    backfill_company(conn)

    rows = conn.execute(
        "SELECT url, title, company, description, location, site, "
        "full_description, application_url, discovered_at, duplicate_of, applied_at "
        "FROM jobs"
    ).fetchall()

    already_marked = sum(1 for r in rows if r["duplicate_of"])
    applied_urls = {r["url"] for r in rows if r["applied_at"]}

    # Build fingerprint groups (only for rows not yet marked as duplicates)
    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    no_fp = 0

    for row in rows:
        if row["duplicate_of"]:
            continue

        fp = fingerprint(row["company"], row["title"])
        if fp is None:
            no_fp += 1
            continue

        groups[fp].append(row)

    # For each group with >1 entry, pick the canonical and mark the rest
    marked = 0
    dup_groups = 0

    for fp, group in groups.items():
        if len(group) < 2:
            continue

        dup_groups += 1
        best = _pick_best(group)
        canonical_url = best["url"]

        for row in group:
            if row["url"] == canonical_url:
                continue
            # Never mark an already-applied job as a duplicate
            if row["url"] in applied_urls:
                continue

            conn.execute(
                "UPDATE jobs SET duplicate_of = ? WHERE url = ? AND duplicate_of IS NULL",
                (canonical_url, row["url"]),
            )
            marked += 1

    conn.commit()

    log.info(
        "Dedup: scanned %d jobs, %d fingerprint groups with duplicates, "
        "%d newly marked, %d previously marked, %d un-fingerprintable",
        len(rows), dup_groups, marked, already_marked, no_fp,
    )

    return {
        "scanned": len(rows),
        "groups": dup_groups,
        "marked": marked,
        "already_marked": already_marked,
        "no_fingerprint": no_fp,
    }


def backfill_company(conn: sqlite3.Connection | None = None) -> int:
    """Backfill the company column from existing data.

    Sources (in priority order):
      1. Workday jobs: site column is the employer name
      2. GitHub repo jobs: title often contains "Company — Title"
      3. JobSpy jobs where site looks like a company name (e.g. "Netflix")

    Returns the number of rows updated.
    """
    if conn is None:
        conn = get_connection()

    updated = 0

    # Workday: site = company name
    cursor = conn.execute(
        "UPDATE jobs SET company = site "
        "WHERE company IS NULL AND strategy = 'workday_api' AND site IS NOT NULL"
    )
    updated += cursor.rowcount

    # GitHub repos: title is "Company — Title"
    rows = conn.execute(
        "SELECT url, title FROM jobs WHERE company IS NULL AND strategy = 'github_repo' AND title LIKE '% — %'"
    ).fetchall()
    for row in rows:
        company = row["title"].split(" — ", 1)[0].strip()
        if company:
            conn.execute("UPDATE jobs SET company = ? WHERE url = ?", (company, row["url"]))
            updated += 1

    conn.commit()
    log.info("Backfilled company for %d jobs", updated)
    return updated


def reset_duplicates(conn: sqlite3.Connection | None = None) -> int:
    """Clear all duplicate_of markers (useful for re-running dedup with new logic).

    Returns the number of rows reset.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.execute(
        "UPDATE jobs SET duplicate_of = NULL WHERE duplicate_of IS NOT NULL"
    )
    conn.commit()
    count = cursor.rowcount
    log.info("Reset %d duplicate markers", count)
    return count
