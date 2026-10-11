"""Score a bounded set of new, deduplicated roles using the configured text backend."""
import json
from datetime import UTC, datetime

from applyagent.applications import filter_applications
from applyagent.config import LOG_DIR, RESUME_PATH, load_env, load_profile
from applyagent.database import get_connection
from applyagent.discovery.quality import evaluate
from applyagent.scoring.scorer import score_job

load_env()
conn = get_connection()
jobs = filter_applications([dict(r) for r in conn.execute(
    'SELECT * FROM jobs WHERE full_description IS NOT NULL AND fit_score IS NULL '
    'AND duplicate_of IS NULL AND applied_at IS NULL')])
jobs = [j for j in jobs if evaluate(j)[0] >= 60 and j.get('application_url')]
jobs.sort(key=lambda j: (-(evaluate(j)[0]), j['title']))
resume = RESUME_PATH.read_text() + '\nConfirmed profile (takes precedence):\n' + json.dumps(load_profile())
results = []
for job in jobs[:20]:
    result = score_job(resume, job)
    results.append({'company': job['company'], 'title': job['title'], 'url': job['url'], **result})
    print(json.dumps(results[-1]), flush=True)
    if not result['score']:
        break
    conn.execute('UPDATE jobs SET fit_score=?, score_reasoning=?, scored_at=? WHERE url=?',
                 (result['score'], result['reasoning'], datetime.now(UTC).isoformat(), job['url']))
    conn.commit()
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / 'queue-scoring.json').write_text(json.dumps(results, indent=2))
