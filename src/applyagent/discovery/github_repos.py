"""GitHub job-repo scraper: parses curated markdown/HTML job tables.

Fetches raw README files from GitHub repos that maintain curated tables of
new-grad SWE positions (speedyapply, SimplifyJobs, vanshb03, etc.) and
stores them via the standard store_jobs() path.

Repo registry lives in config/github_repos.yaml — add new repos there
without any code changes.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone

import httpx
import yaml
from bs4 import BeautifulSoup

from applyagent import config
from applyagent.config import CONFIG_DIR
from applyagent.database import get_connection, init_db, store_jobs

log = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file}"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_repo_config() -> dict:
    """Load GitHub repo registry from ~/.applyagent/github_repos.yaml, parsing fallback if missing."""
    path = config.GITHUB_REPOS_PATH
    if not path.exists():
        # Fallback to package config if user hasn't generated it yet
        path = CONFIG_DIR / "github_repos.yaml"
        if not path.exists():
            log.warning("github_repos.yaml not found at %s", path)
            return {}
            
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("repos", {})


# ---------------------------------------------------------------------------
# Location filtering (reuse pattern from workday.py)
# ---------------------------------------------------------------------------

def _load_location_filter() -> tuple[list[str], list[str]]:
    search_cfg = config.load_search_config()
    loc = search_cfg.get("location", {})
    accept = loc.get("accept_patterns", search_cfg.get("location_accept", []))
    reject = loc.get("reject_patterns", search_cfg.get("location_reject_non_remote", []))
    return accept, reject


def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    """Check if a job location passes the user's location filter."""
    if not location:
        return True
    loc = location.lower()
    if any(r in loc for r in ("remote", "anywhere", "work from home", "wfh", "distributed")):
        return True
    for r in reject:
        if r.lower() in loc:
            return False
    for a in accept:
        if a.lower() in loc:
            return True
    return False


# ---------------------------------------------------------------------------
# Cell extraction helpers
# ---------------------------------------------------------------------------

_HREF_RE = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)
_STRONG_RE = re.compile(r'<strong>([^<]+)</strong>', re.IGNORECASE)
_ANCHOR_TEXT_RE = re.compile(r'<a\s[^>]*>([^<]+)</a>', re.IGNORECASE)


def _extract_apply_url(cell: str) -> str | None:
    """Extract the direct-apply URL from an apply cell, skipping simplify.jobs redirects."""
    matches = _HREF_RE.findall(cell)
    for url in matches:
        if "simplify.jobs" in url:
            continue
        return url
    # Fall back to the first link if all are simplify
    return matches[0] if matches else None


def _is_closed(cell: str) -> bool:
    """Detect closed postings — marked with lock emoji."""
    return "\U0001f512" in cell  # 🔒


def _extract_company(cell: str) -> str:
    """Pull company name from <strong> or <a> tags, or fall back to plain text."""
    # Try <strong>Company</strong> first
    m = _STRONG_RE.search(cell)
    if m:
        # The strong text might itself contain an <a> tag; strip it
        inner = m.group(1).strip()
        am = _ANCHOR_TEXT_RE.search(inner)
        return am.group(1).strip() if am else inner

    # Try <a href="...">Company</a>
    m = _ANCHOR_TEXT_RE.search(cell)
    if m:
        return m.group(1).strip()

    # Try **Company** (markdown bold)
    m = re.search(r'\*\*([^*]+)\*\*', cell)
    if m:
        return m.group(1).strip()

    # Plain text fallback — strip HTML tags
    return re.sub(r'<[^>]+>', '', cell).strip()


def _clean_location(cell: str) -> str:
    """Expand <details> tags and normalise HTML to readable text."""
    # Expand <details><summary>N locations</summary>City1<br>City2</details>
    details = re.search(
        r'<details>\s*<summary>[^<]*</summary>(.*?)</details>',
        cell, re.IGNORECASE | re.DOTALL,
    )
    if details:
        inner = details.group(1)
    else:
        inner = cell

    # Replace <br>, </br>, <br/> with ", "
    inner = re.sub(r'</?br\s*/?>', ', ', inner, flags=re.IGNORECASE)
    # Strip remaining HTML
    inner = re.sub(r'<[^>]+>', '', inner)
    # Collapse whitespace
    inner = re.sub(r'\s+', ' ', inner).strip()
    # Clean up doubled commas / leading/trailing commas
    inner = re.sub(r',\s*,', ',', inner)
    inner = inner.strip(', ')
    return inner


def _strip_html(text: str) -> str:
    """Remove all HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text).strip()


# ---------------------------------------------------------------------------
# Markdown pipe-table parser
# ---------------------------------------------------------------------------

def _parse_markdown_pipe(text: str, section_cfg: dict) -> list[dict]:
    """Parse a markdown pipe table section into job dicts."""
    cols = section_cfg["columns"]
    start = section_cfg.get("start_marker")
    end = section_cfg.get("end_marker")

    # Slice to section if markers are specified
    if start:
        idx = text.find(start)
        if idx == -1:
            log.debug("Start marker %r not found", start)
            return []
        text = text[idx + len(start):]
    if end:
        idx = text.find(end)
        if idx != -1:
            text = text[:idx]

    jobs: list[dict] = []
    in_table = False

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break  # end of table
            continue

        # Skip separator rows (|---|---|...)
        if re.match(r'^\|[\s\-:]+\|', line):
            in_table = True
            continue

        # Skip header row (first pipe row before separator)
        if not in_table:
            continue

        cells = [c.strip() for c in line.split("|")]
        # split on | gives empty strings at start/end
        cells = [c for c in cells if c or cells.index(c) not in (0, len(cells) - 1)]
        # More robust: just strip first and last empty
        raw_cells = line.split("|")
        if raw_cells and raw_cells[0].strip() == "":
            raw_cells = raw_cells[1:]
        if raw_cells and raw_cells[-1].strip() == "":
            raw_cells = raw_cells[:-1]
        cells = [c.strip() for c in raw_cells]

        apply_idx = cols.get("apply")
        if apply_idx is not None and apply_idx < len(cells):
            apply_cell = cells[apply_idx]
            if _is_closed(apply_cell):
                continue
            url = _extract_apply_url(apply_cell)
        else:
            url = None

        if not url:
            continue

        company_idx = cols.get("company")
        title_idx = cols.get("title")
        location_idx = cols.get("location")
        salary_idx = cols.get("salary")

        company = _extract_company(cells[company_idx]) if company_idx is not None and company_idx < len(cells) else ""
        title = _strip_html(cells[title_idx]) if title_idx is not None and title_idx < len(cells) else ""
        location = _clean_location(cells[location_idx]) if location_idx is not None and location_idx < len(cells) else ""
        salary = _strip_html(cells[salary_idx]) if salary_idx is not None and salary_idx < len(cells) else None

        full_title = f"{company} — {title}" if company and title else title or company

        jobs.append({
            "url": url,
            "title": full_title,
            "company": company or None,
            "salary": salary,
            "location": location,
            "description": title,
        })

    return jobs


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------

def _parse_html_table(text: str, columns: dict) -> list[dict]:
    """Parse an HTML <table> into job dicts using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    table = soup.find("table")
    if not table:
        log.warning("No <table> found in HTML content")
        return []

    jobs: list[dict] = []
    tbody = table.find("tbody") or table

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        apply_idx = columns.get("apply")
        if apply_idx is not None and apply_idx < len(cells):
            apply_html = str(cells[apply_idx])
            if _is_closed(apply_html) or _is_closed(cells[apply_idx].get_text()):
                continue
            url = _extract_apply_url(apply_html)
        else:
            url = None

        if not url:
            continue

        company_idx = columns.get("company")
        title_idx = columns.get("title")
        location_idx = columns.get("location")
        salary_idx = columns.get("salary")

        company = _extract_company(str(cells[company_idx])) if company_idx is not None and company_idx < len(cells) else ""
        title = cells[title_idx].get_text(strip=True) if title_idx is not None and title_idx < len(cells) else ""
        location = _clean_location(str(cells[location_idx])) if location_idx is not None and location_idx < len(cells) else ""
        salary = cells[salary_idx].get_text(strip=True) if salary_idx is not None and salary_idx < len(cells) else None

        full_title = f"{company} — {title}" if company and title else title or company

        jobs.append({
            "url": url,
            "title": full_title,
            "company": company or None,
            "salary": salary,
            "location": location,
            "description": title,
        })

    return jobs


# ---------------------------------------------------------------------------
# Fetch + parse one repo
# ---------------------------------------------------------------------------

def _process_repo(
    repo_key: str,
    repo_cfg: dict,
    accept_locs: list[str],
    reject_locs: list[str],
) -> dict:
    """Fetch and parse one GitHub repo. Returns stats dict."""
    owner = repo_cfg["owner"]
    repo = repo_cfg["repo"]
    branch = repo_cfg.get("branch", "main")
    file = repo_cfg.get("file", "README.md")

    url = RAW_URL.format(owner=owner, repo=repo, branch=branch, file=file)
    log.info("%s/%s: fetching %s", owner, repo, url)

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        log.error("%s/%s: fetch failed: %s", owner, repo, e)
        return {"repo": f"{owner}/{repo}", "found": 0, "new": 0, "existing": 0, "error": str(e)}

    fmt = repo_cfg.get("format", "markdown_pipe")
    all_jobs: list[dict] = []

    if fmt == "html_table":
        columns = repo_cfg.get("columns", {})
        all_jobs = _parse_html_table(text, columns)
    elif fmt == "markdown_pipe":
        sections = repo_cfg.get("sections", [])
        if not sections:
            # Treat the whole file as one section
            sections = [{"name": "main", "columns": repo_cfg.get("columns", {})}]
        for sec in sections:
            parsed = _parse_markdown_pipe(text, sec)
            log.info("%s/%s [%s]: parsed %d jobs", owner, repo, sec.get("name", "?"), len(parsed))
            all_jobs.extend(parsed)
    else:
        log.warning("%s/%s: unknown format %r", owner, repo, fmt)
        return {"repo": f"{owner}/{repo}", "found": 0, "new": 0, "existing": 0}

    # Location filter
    if accept_locs or reject_locs:
        before = len(all_jobs)
        all_jobs = [j for j in all_jobs if _location_ok(j.get("location"), accept_locs, reject_locs)]
        filtered = before - len(all_jobs)
        if filtered:
            log.info("%s/%s: filtered %d jobs by location (%d remaining)", owner, repo, filtered, len(all_jobs))

    if not all_jobs:
        log.info("%s/%s: no jobs after filtering", owner, repo)
        return {"repo": f"{owner}/{repo}", "found": 0, "new": 0, "existing": 0}

    # Store via standard path
    conn = get_connection()
    site = repo_cfg.get("repo", repo_key)
    new, existing = store_jobs(conn, all_jobs, site=site, strategy="github_repo")
    log.info("%s/%s: %d new, %d duplicates", owner, repo, new, existing)

    return {"repo": f"{owner}/{repo}", "found": len(all_jobs), "new": new, "existing": existing}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_github_repos(workers: int = 1) -> dict:
    """Scrape all configured GitHub job repos.

    Args:
        workers: Unused (kept for interface parity with other discovery sources).

    Returns:
        Dict with aggregate stats: found, new, existing, repos.
    """
    repos = _load_repo_config()
    if not repos:
        log.warning("No GitHub repos configured. Check config/github_repos.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "repos": 0}

    init_db()
    accept_locs, reject_locs = _load_location_filter()

    total_found = 0
    total_new = 0
    total_existing = 0
    errors = 0

    for key, cfg in repos.items():
        result = _process_repo(key, cfg, accept_locs, reject_locs)
        total_found += result["found"]
        total_new += result["new"]
        total_existing += result["existing"]
        if "error" in result:
            errors += 1

    log.info("GitHub repos done: %d found, %d new, %d existing, %d errors across %d repos",
             total_found, total_new, total_existing, errors, len(repos))

    return {
        "found": total_found,
        "new": total_new,
        "existing": total_existing,
        "repos": len(repos),
        "errors": errors,
    }
