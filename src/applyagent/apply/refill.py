"""Replenish the live application queue without relaxing candidate fit."""
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime

from applyagent import config
from applyagent.applications import filter_applications
from applyagent.database import get_connection
from applyagent.discovery.quality import JUNIOR, evaluate, has_early_experience


def replenish(min_score: int = 7, batch_size: int = 20,
              max_candidates: int = 100, discover: bool = True,
              deadline: float | None = None) -> int:
    from applyagent.dedup import run_dedup
    from applyagent.enrichment.detail import scrape_site_batch
    from applyagent.scheduled import run_scheduled
    from applyagent.scoring.scorer import score_job

    if batch_size < 1 or max_candidates < 1:
        raise ValueError('Refill batch and candidate limits must be positive')
    conn = get_connection()
    from applyagent.review import _load, deferred_reason
    review_state = _load()
    max_per_company = config.load_search_config().get("quality", {}).get("max_per_company", 3)
    if type(max_per_company) is not int or max_per_company < 1:
        raise ValueError("quality.max_per_company must be positive")

    def candidates():
        rows = conn.execute(
            "SELECT * FROM jobs WHERE duplicate_of IS NULL AND applied_at IS NULL "
            "AND apply_status IS NULL AND fit_score IS NULL "
            "AND ((NULLIF(TRIM(full_description), '') IS NOT NULL "
            "AND NULLIF(TRIM(application_url), '') IS NOT NULL) OR detail_scraped_at IS NULL)"
        ).fetchall()
        jobs = filter_applications([dict(row) for row in rows])
        jobs = [job for job in jobs if not deferred_reason(job, state=review_state)]
        ranked = sorted((j for j in jobs if evaluate(j)[0] >= 60),
                      key=lambda j: (not (bool(JUNIOR.search(j.get('title') or '')) or has_early_experience(j)),
                                     bool(re.search('founding', j.get('title') or '', re.IGNORECASE)),
                                     -evaluate(j)[0], j['url']))
        counts = Counter()
        diverse = []
        for job in ranked:
            company = re.sub(r'[^a-z0-9]', '', (job.get('company') or '').casefold()) or job['url']
            if counts[company] < max_per_company:
                diverse.append(job)
                counts[company] += 1
        return diverse

    resume = config.RESUME_PATH.read_text() + '\nConfirmed profile (takes precedence):\n' + json.dumps(config.load_profile())
    prepared = 0
    examined = set()
    while len(examined) < max_candidates:
        if deadline is not None and time.monotonic() >= deadline:
            break
        jobs = [job for job in candidates() if job['url'] not in examined]
        if not jobs:
            if not discover or prepared:
                break
            # At most one discovery per replenishment. Do not repeatedly hammer
            # blocked sources when no candidates can be prepared.
            run_scheduled(timeout=min(600, max(1, int(deadline - time.monotonic()))) if deadline else 600)
            run_dedup(conn)
            discover = False
            continue
        for job in jobs[:min(batch_size, max_candidates - len(examined))]:
            if deadline is not None and time.monotonic() >= deadline:
                return prepared
            examined.add(job['url'])
            if not job.get('full_description') or not job.get('application_url'):
                scrape_site_batch(conn, job['site'], [(job['url'], job['title'], job.get('application_url'))], delay=0)
                job = dict(conn.execute('SELECT * FROM jobs WHERE url=?', (job['url'],)).fetchone())
            if not job.get('full_description') or not job.get('application_url'):
                continue
            result = score_job(resume, job)
            if not result['score']:
                # Preserve the unscored candidate, but don't describe a broken
                # harness as an ordinary empty queue or successful run.
                raise RuntimeError('Scoring unavailable: ' + result['reasoning'])
            ready = result['score'] >= min_score
            conn.execute(
                'UPDATE jobs SET fit_score=?, score_reasoning=?, scored_at=?, '
                'tailored_resume_path=COALESCE(tailored_resume_path, ?) WHERE url=?',
                (result['score'], result['reasoning'], datetime.now(UTC).isoformat(),
                 str(config.RESUME_PDF_PATH) if ready else None, job['url']))
            conn.commit()
            prepared += int(ready)
        if prepared:
            break
    return prepared
