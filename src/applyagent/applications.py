"""Private, evidence-backed application history for discovery reports."""

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from applyagent import config
from applyagent.discovery.quality import canonical_url


def application_identity(url: str) -> str:
    parts = urlsplit(url)
    match = re.fullmatch(r'/[^/]+/([0-9a-fA-F-]{36})(?:/application)?/?', parts.path)
    if parts.hostname == 'jobs.ashbyhq.com' and match:
        return 'ashby:' + match[1].lower()
    match = re.fullmatch(r'/([^/]+)/([0-9a-fA-F-]{36})(?:/apply)?/?', parts.path)
    if parts.hostname in {'jobs.lever.co', 'jobs.eu.lever.co'} and match:
        return f'lever:{parts.hostname}:{match[1].lower()}:{match[2].lower()}'
    if parts.hostname in {'boards.greenhouse.io', 'job-boards.greenhouse.io',
                          'boards.eu.greenhouse.io', 'job-boards.eu.greenhouse.io'}:
        match = re.fullmatch(r'/([^/]+)/jobs/(\d+)/?', parts.path)
        if match:
            return f'greenhouse:{match[1].lower()}:{match[2]}'
        query = parse_qs(parts.query)
        if parts.path == '/embed/job_app' and query.get('for') and query.get('token'):
            return f"greenhouse:{query['for'][0].lower()}:{query['token'][0]}"
    return canonical_url(url)


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def company_key(value: str) -> str:
    value = re.sub(r"\b(inc|llc|corporation)\b", "", normalized(value)).strip()
    return {"clay labs": "clay", "palantir technologies": "palantir"}.get(value, value)


def filter_applications(jobs: list[dict], path: Path | None = None) -> list[dict]:
    """Suppress exact role matches; retain distinct roles with a history flag.

    Unknown titles never match every opening at an employer. Historical roles
    remain context only, since a new recruiting cycle is a new opportunity.
    Invalid registry data raises instead of silently recommending duplicates.
    """
    path = path or config.APP_DIR / "already_applied.json"
    if not path.exists():
        return jobs
    records = json.loads(path.read_text())["applications"]
    submitted_targets = {
        application_identity(url)
        for record in records if record.get('submission_confirmed')
        for url in [record.get('url'), record.get('application_url'),
                    *(e.get('url') for e in record.get('evidence', []))]
        if url
    }
    result = []
    for job in jobs:
        if job.get("applied_at"):
            continue
        if any(application_identity(url) in submitted_targets
               for url in (job.get('url'), job.get('application_url')) if url):
            continue
        history = [r for r in records if company_key(r["company"]) == company_key(job.get("company", ""))]
        title = normalized(job.get("title", ""))
        if any(r.get("scope") != "historical" and r.get("role")
               and title in {normalized(r["role"]), *(normalized(t) for t in r.get("role_aliases", []))}
               for r in history):
            continue
        item = dict(job)
        if history:
            item["application_history"] = [
                {k: r.get(k) for k in ("company", "role", "status", "scope", "submission_confirmed")}
                for r in history
            ]
            item["application_review_note"] = "Prior application/recruiting history at this company; verify this is a different opening before applying."
        result.append(item)
    return result
