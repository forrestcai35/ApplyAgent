"""Private, evidence-backed application history for discovery reports."""

import json
import re
from pathlib import Path

from applyagent import config


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
    result = []
    for job in jobs:
        if job.get("applied_at"):
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
