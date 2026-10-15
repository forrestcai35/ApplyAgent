import json

import pytest

from applyagent import config
from applyagent.apply import receipts
from applyagent.database import init_db

URL = 'https://jobs.example/thank-you'
MESSAGE = 'Your application was successfully submitted.'
FINAL = f'CONFIRMATION_URL: {URL}\nCONFIRMATION_MESSAGE: {MESSAGE}\nRESULT:APPLIED'


def write_log(path, browser=True):
    items = []
    if browser:
        items.append({'type': 'mcp_tool_call', 'server': 'playwright', 'status': 'completed',
                      'result': {'content': [{'type': 'text', 'text': f'{URL}\n{MESSAGE}'}]}})
    items.append({'type': 'agent_message', 'text': FINAL})
    path.write_text('\n'.join(json.dumps({'type': 'item.completed', 'item': i}) for i in items))


def test_final_claim_alone_is_not_confirmation(tmp_path):
    log = tmp_path / 'run.jsonl'
    write_log(log, browser=False)
    with pytest.raises(ValueError, match='browser output'):
        receipts.confirmed_evidence(log)
    write_log(log)
    assert receipts.confirmed_evidence(log)['confirmation'] == MESSAGE


def test_receipt_recovery_is_idempotent_and_updates_daily_count(monkeypatch, tmp_path):
    conn = init_db(tmp_path / 'jobs.db')
    resume = tmp_path / 'resume.pdf'
    resume.write_bytes(b'private resume')
    conn.execute('INSERT INTO jobs (url,company,title,apply_status,applied_at,tailored_resume_path) '
                 'VALUES (?,?,?,?,?,?)',
                 ('job', 'Employer', 'Engineer', 'applied', '2026-10-11T20:00:00+00:00', str(resume)))
    conn.commit()
    monkeypatch.setattr(receipts, 'get_connection', lambda: conn)
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(config, 'load_profile', dict)
    log = tmp_path / 'run.jsonl'
    write_log(log)
    receipt = receipts.record_submission('job', log)
    assert receipt.exists()
    receipts.record_submission('job', log)
    registry = json.loads((tmp_path / 'already_applied.json').read_text())
    daily = json.loads((tmp_path / 'reports/applications-2026-10-11.json').read_text())
    assert len(registry['applications']) == 1
    assert daily['confirmed_submissions'] == 1
    assert daily['remaining_daily_capacity'] == 9


def test_receipt_failure_never_turns_submission_into_retry(monkeypatch):
    from unittest.mock import Mock

    from applyagent.apply import launcher

    launcher._stop_event.clear()
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kw: {
        'url': 'job', 'title': 'Engineer', '_confirmation_log': 'saved.jsonl'})
    monkeypatch.setattr(launcher, 'launch_chrome', Mock())
    monkeypatch.setattr(launcher, 'cleanup_worker', Mock())
    monkeypatch.setattr(launcher, 'update_state', Mock())
    monkeypatch.setattr(launcher, 'add_event', Mock())
    monkeypatch.setattr(launcher, 'run_job', lambda *a, **kw: ('applied', 100))
    marked = Mock()
    monkeypatch.setattr(launcher, 'mark_result', marked)
    monkeypatch.setattr(receipts, 'record_submission', Mock(side_effect=OSError('disk full')))
    assert launcher.worker_loop(until_submitted=1) == (1, 0)
    marked.assert_called_once_with('job', 'applied', duration_ms=100)
