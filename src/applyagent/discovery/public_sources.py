"""Browser-free RSS and public HTML JobPosting collectors."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from applyagent.database import get_connection, store_jobs
from applyagent.discovery.quality import canonical_url, evaluate

log = logging.getLogger(__name__)


def _expired(value: str | None) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        date = datetime.fromisoformat(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            return False
    if date.tzinfo is None:
        date = date.replace(tzinfo=UTC)
    return date < datetime.now(UTC)


def parse_feed(text: str, source: dict) -> list[dict]:
    root = ElementTree.fromstring(text)
    jobs = []
    for item in root.findall(".//item"):
        if _expired(item.findtext("expires_at")):
            continue
        title = item.findtext("title") or ""
        company = None
        if source.get("company_prefix") and ": " in title:
            company, title = title.split(": ", 1)
        location = item.findtext("region") or item.findtext("location") or ""
        if source.get("remote"):
            location = f"Remote — {location}".strip(" —")
        url = item.findtext("link") or ""
        if not url.startswith(("https://", "http://")):
            continue
        jobs.append({"url": canonical_url(url), "title": title, "company": company,
                     "location": location, "description": BeautifulSoup(
                         item.findtext("description") or "", "html.parser").get_text(" ", strip=True)})
    return jobs


def _walk(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        if "JobPosting" in (types if isinstance(types, list) else [types]):
            yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _walk(item)


def parse_page(text: str, url: str) -> list[dict]:
    jobs = []
    for script in BeautifulSoup(text, "html.parser").find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text())
        except (ValueError, TypeError):
            continue
        for node in _walk(data):
            if _expired(node.get("validThrough")):
                continue
            places = node.get("jobLocation", [])
            if isinstance(places, dict):
                places = [places]
            locations = []
            for place in places:
                address = place.get("address", {}) if isinstance(place, dict) else {}
                if isinstance(address, dict):
                    locations.append(", ".join(str(address[k]) for k in ("addressLocality", "addressRegion") if address.get(k)))
            if node.get("jobLocationType") == "TELECOMMUTE":
                locations.insert(0, "Remote")
            employer = node.get("hiringOrganization") or {}
            target = node.get("url") or url
            if not isinstance(target, str) or not target.startswith(("https://", "http://")):
                continue
            jobs.append({"url": canonical_url(target), "title": node.get("title"),
                         "company": employer.get("name") if isinstance(employer, dict) else None,
                         "location": "; ".join(locations), "description": BeautifulSoup(
                             node.get("description") or "", "html.parser").get_text(" ", strip=True)})
    return jobs


def run_public_sources(cfg: dict) -> dict:
    results = []
    with httpx.Client(timeout=25, follow_redirects=True) as client:
        for kind, sources in (("rss", cfg.get("feeds", [])), ("json_ld", cfg.get("public_pages", []))):
            for source in sources:
                name = source.get("name", source["url"])
                try:
                    response = client.get(source["url"])
                    response.raise_for_status()
                    jobs = parse_feed(response.text, source) if kind == "rss" else parse_page(response.text, source["url"])
                    if not jobs:
                        results.append({"source": name, "new": 0, "status": "empty_or_unreadable"})
                        continue
                    found = len(jobs)
                    if cfg.get("quality", {}).get("enabled"):
                        jobs = [job for job in jobs if evaluate(job)[0]]
                    new, existing = store_jobs(get_connection(), jobs, site=name, strategy=kind)
                    results.append({"source": name, "found": found, "eligible": len(jobs), "new": new,
                                    "existing": existing, "status": "ok"})
                except (httpx.HTTPError, ElementTree.ParseError, ValueError, TypeError):
                    log.exception("Public source failed: %s", name)
                    results.append({"source": name, "new": 0, "status": "error"})
    return {"new": sum(r["new"] for r in results), "errors": sum(r["status"] != "ok" for r in results),
            "sources": results}
