"""GET-only public employer postings from Greenhouse, Lever and Ashby."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from applyagent import config
from applyagent.applications import application_identity, filter_applications
from applyagent.database import get_connection, store_jobs
from applyagent.discovery.backoff import BoardBackoff
from applyagent.discovery.quality import canonical_url, evaluate
from applyagent.review import deferred_reason

log = logging.getLogger(__name__)


def endpoint(source: dict) -> str:
    board = source['board']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', board):
        raise ValueError('Invalid public employer board token')
    provider = source['provider']
    if provider == 'greenhouse':
        return f'https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true'
    if provider == 'lever':
        return f'https://api.lever.co/v0/postings/{board}?mode=json'
    if provider == 'ashby':
        return f'https://api.ashbyhq.com/posting-api/job-board/{board}?includeCompensation=true'
    raise ValueError('Supported employer APIs: greenhouse, lever, ashby')


def _text(value: str) -> str:
    return BeautifulSoup(html.unescape(html.unescape(value or '')), 'html.parser').get_text('\n', strip=True)


def parse_postings(payload, source: dict) -> list[dict]:
    provider = source['provider']
    if provider != 'lever' and not isinstance(payload, dict):
        raise TypeError('Invalid public postings response')
    rows = payload if provider == 'lever' else payload.get('jobs', [])
    if not isinstance(rows, list):
        raise TypeError('Invalid public postings response')
    jobs = []
    for row in rows:
        if provider == 'greenhouse':
            if row.get('internal_job_id', True) is None:
                continue  # Prospect/talent-pool posts are not distinct open jobs.
            title = row.get('title')
            location = (row.get('location') or {}).get('name', '')
            description = _text(row.get('content', ''))
            url = row.get('absolute_url')
            apply_url = url
            salary = None
        elif provider == 'lever':
            title = row.get('text')
            categories = row.get('categories') or {}
            locations = categories.get('allLocations') or [categories.get('location', '')]
            location = '; '.join(locations)
            if row.get('workplaceType') == 'remote':
                location = 'Remote; ' + location
            description = '\n'.join([_text(row.get('description', '')),
                *(_text(item.get('text', '') + '\n' + item.get('content', '')) for item in row.get('lists', [])),
                _text(row.get('additional', ''))])
            url = row.get('hostedUrl')
            apply_url = row.get('applyUrl') or url
            salary = None
        elif provider == 'ashby':
            if row.get('isListed') is False:
                continue
            title = row.get('title')
            locations = [row.get('location', '')]
            locations += [place.get('location', '') for place in row.get('secondaryLocations', [])]
            location = '; '.join(locations)
            workplace = (row.get('workplaceType') or '').lower()
            if workplace == 'remote' or (not workplace and row.get('isRemote') is True):
                location = 'Remote; ' + location
            description = row.get('descriptionPlain') or _text(row.get('descriptionHtml', ''))
            url = row.get('jobUrl')
            apply_url = row.get('applyUrl') or url
            salary = (row.get('compensation') or {}).get('scrapeableCompensationSalarySummary')
        else:
            raise ValueError('Unsupported employer API')
        if not title or not isinstance(url, str) or not url.startswith('https://'):
            continue
        if not isinstance(apply_url, str) or not apply_url.startswith('https://'):
            continue
        jobs.append({'url': canonical_url(url), 'application_url': canonical_url(apply_url),
                     'title': title, 'company': source['name'], 'location': location,
                     'description': description, 'full_description': description,
                     'salary': salary, 'strategy': 'employer_api'})
    return jobs


def run_employer_apis(cfg: dict) -> dict:
    settings = cfg.get('employer_api_settings', {})
    deadline = time.monotonic() + settings.get('max_seconds', 60)
    results = []
    conn = get_connection()
    backoff = BoardBackoff()
    boards = cfg.get('employer_boards', [])
    cursor_path = config.APP_DIR / 'employer-source-cursor.json'
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    keys = [f"{source['provider']}:{source['board']}" for source in boards]
    start = keys.index(cursor['next_board']) if cursor.get('next_board') in keys else 0
    indices = [(start + offset) % len(boards) for offset in
               range(min(settings.get('max_boards', 15), len(boards)))]
    sources = [(index, boards[index]) for index in indices]
    with httpx.Client(follow_redirects=True) as client:
        for index, source in sources:
            if time.monotonic() >= deadline:
                results.append({'source': source['name'], 'status': 'budget_reached', 'new': 0})
                break
            # Advance even after a source error/backoff so later boards are not
            # starved. A budget stop above leaves the next board unconsumed.
            config.APP_DIR.mkdir(parents=True, exist_ok=True)
            temporary = cursor_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'next_board': keys[(index + 1) % len(boards)]}) + '\n')
            temporary.chmod(0o600)
            temporary.replace(cursor_path)
            key = f"ats:{source['provider']}:{source['board']}"
            if not backoff.allows(key):
                results.append({'source': source['name'], 'provider': source['provider'], 'status': 'backoff', 'new': 0})
                continue
            try:
                response = client.get(endpoint(source), timeout=min(8, max(0.1, deadline - time.monotonic())))
                response.raise_for_status()
                jobs = parse_postings(response.json(), source)
                found = len(jobs)
                aliases = conn.execute('SELECT url,application_url FROM jobs').fetchall()
                targets = {application_identity(row['application_url'] or row['url']): row['url'] for row in aliases}
                # Refresh normalized location for existing rows even when the
                # corrected location now falls outside the discovery funnel.
                for job in jobs:
                    target = targets.get(application_identity(job['application_url']))
                    if target:
                        conn.execute('UPDATE jobs SET location=?,full_description=? WHERE url=?',
                                     (job['location'], job['description'], target))
                conn.commit()
                jobs = [job for job in jobs if evaluate(job)[0] >= cfg.get('quality', {}).get('min_score', 60)]
                fits = len(jobs)
                jobs = filter_applications(jobs)
                jobs = [job for job in jobs if not deferred_reason(job)]
                # Public APIs provide employer descriptions and application URLs;
                # save them directly instead of launching enrichment browsers.
                unseen = [job for job in jobs if application_identity(job['application_url']) not in targets]
                new, same_url = store_jobs(conn, unseen, site=source['provider'], strategy='employer_api')
                existing = len(jobs) - len(unseen) + same_url
                for job in jobs:
                    identity = application_identity(job['application_url'])
                    target = targets.get(identity, job['url'])
                    conn.execute('UPDATE jobs SET full_description=?,application_url=?,detail_scraped_at=?,detail_error=NULL '
                                 'WHERE url=?', (job['description'], job['application_url'], datetime.now(UTC).isoformat(), target))
                conn.commit()
                results.append({'source': source['name'], 'provider': source['provider'], 'status': 'ok',
                                'found': found, 'fit_candidates': fits, 'eligible_after_history_holds': len(jobs),
                                'new': new, 'existing': existing})
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
                backoff.failed(key, str(error))
                log.warning('Public employer board failed: %s (%s)', source['name'], type(error).__name__)
                results.append({'source': source['name'], 'provider': source['provider'], 'status': 'error', 'new': 0})
    return {'new': sum(item.get('new', 0) for item in results),
            'errors': sum(item['status'] != 'ok' for item in results), 'sources': results}
