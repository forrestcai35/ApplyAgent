"""Verify browser evidence and persist local, duplicate-safe submission receipts."""
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from applyagent import config
from applyagent.database import get_connection


def confirmed_evidence(log_path: Path, final: str | None = None) -> dict:
    from applyagent.apply.backend import parse_result

    items = []
    for line in log_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'item.completed':
            items.append(event.get('item', {}))
    if final is None:
        final = next((i['text'] for i in reversed(items) if i.get('type') == 'agent_message'), '')
    if parse_result(final) != 'applied':
        raise ValueError('No unambiguous final submission confirmation')
    fields = dict(line.split(': ', 1) for line in final.splitlines() if ': ' in line)
    url, message = fields['CONFIRMATION_URL'], fields['CONFIRMATION_MESSAGE']
    # Require both values in the same successful browser result, not just model prose.
    if not any(
        i.get('type') == 'mcp_tool_call' and i.get('server') == 'playwright'
        and i.get('status') == 'completed' and not i.get('error')
        and url in json.dumps(i.get('result', {}), ensure_ascii=False)
        and message in json.dumps(i.get('result', {}), ensure_ascii=False)
        for i in items
    ):
        raise ValueError('Confirmation is not supported by observed browser output')
    return {'url': url, 'confirmation': message, 'log': str(log_path.resolve())}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_submission(job_url: str, log_path: Path) -> Path:
    row = get_connection().execute('SELECT * FROM jobs WHERE url=?', (job_url,)).fetchone()
    if row is None or row['apply_status'] != 'applied' or not row['applied_at']:
        raise ValueError('Database does not contain a confirmed submission')
    job = dict(row)
    evidence = confirmed_evidence(log_path)
    resume = Path(job['tailored_resume_path'])
    day = datetime.fromisoformat(job['applied_at']).astimezone(ZoneInfo('America/Los_Angeles')).date()
    slug = re.sub(r'[^a-z0-9]+', '-', (job['company'] or 'employer').lower()).strip('-')
    receipt = config.APP_DIR / 'reports' / f'{slug}-submitted-{day}-{hashlib.sha256(job_url.encode()).hexdigest()[:8]}.json'
    entry = {
        'company': job['company'], 'role': job['title'], 'location': job['location'],
        'url': job_url, 'status': 'submitted', 'submission_confirmed': True,
        'submitted_at': job['applied_at'], 'scope': str(day.year),
        'resume_path': str(resume), 'resume_sha256': hashlib.sha256(resume.read_bytes()).hexdigest(),
        'evidence': [evidence], 'receipt': str(receipt),
    }
    config.APP_DIR.mkdir(parents=True, exist_ok=True)
    with (config.APP_DIR / '.receipt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_json(receipt, entry)
        paths = (config.APP_DIR / 'already_applied.json',
                 config.APP_DIR / 'reports' / f'applications-{day}.json')
        for path in paths:
            data = json.loads(path.read_text()) if path.exists() else {
                'date': str(day), 'timezone': 'America/Los_Angeles', 'applications': []}
            if not any(a.get('url') == job_url and a.get('submission_confirmed') for a in data['applications']):
                data['applications'].append(entry)
            data['updated_at'] = datetime.now(UTC).isoformat()
            if path.name.startswith('applications-'):
                count = sum(bool(a.get('submission_confirmed')) for a in data['applications'])
                cap = config.load_profile().get('application_workflow', {}).get('daily_submission_cap', 10)
                data.update(confirmed_submissions=count, remaining_daily_capacity=max(0, cap-count))
            _write_json(path, data)
    return receipt
