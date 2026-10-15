import json
from datetime import datetime, timezone

import pytest

from applyagent import config
from applyagent.apply import runtime
from applyagent.database import init_db


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(config, 'load_profile', lambda: {
        'application_workflow': {'daily_submission_cap': 10, 'timezone': 'America/Los_Angeles'}})
    (tmp_path / 'reports').mkdir()
    return tmp_path


NOW = datetime(2026, 10, 13, 18, tzinfo=timezone.utc)
URL = 'https://jobs.ashbyhq.com/employer/12345678-1234-1234-1234-123456789abc'


def entry(url=URL, timestamp='2026-10-13T08:00:00+00:00', confirmed=True):
    return {'company': 'Employer', 'role': 'Junior Engineer', 'url': url,
            'submitted_at': timestamp, 'submission_confirmed': confirmed}


def test_cap_unions_registry_receipts_ledger_and_database_without_double_counting(data):
    (data / 'already_applied.json').write_text(json.dumps({'applications': [entry()]}))
    receipt = entry(URL + '/application?utm_source=board')
    (data / 'reports/employer-submitted-2026-10-13.json').write_text(json.dumps(receipt))
    (data / 'reports/applications-2026-10-13.json').write_text(json.dumps({
        'applications': [receipt, entry('https://other.example/job')], 'confirmed_submissions': 2}))
    conn = init_db(data / 'applyagent.db')
    conn.execute('INSERT INTO jobs (url, application_url, title, company, apply_status, applied_at) '
                 'VALUES (?, ?, ?, ?, ?, ?)',
                 ('https://board.example/post', URL, 'Junior Engineer', 'Employer',
                  'applied', '2026-10-13T09:00:00+00:00'))
    conn.execute('INSERT INTO jobs (url, title, company, apply_status, applied_at) '
                 'VALUES (?, ?, ?, ?, ?)',
                 ('https://third.example/job', 'Engineer I', 'Third', 'applied', '2026-10-13T09:00:00+00:00'))
    conn.commit()
    budget = runtime.daily_budget(NOW)
    assert budget.date == '2026-10-13'
    assert budget.submitted == 3
    assert budget.remaining == 7


def test_calendar_day_is_pacific_and_uncertain_receipts_do_not_count(data):
    entries = [entry('https://prior.example/job', '2026-10-13T06:59:59+00:00'),
               entry('https://today.example/job', '2026-10-13T07:00:00+00:00'),
               entry('https://uncertain.example/job', confirmed=False)]
    (data / 'already_applied.json').write_text(json.dumps({'applications': entries}))
    assert runtime.daily_budget(NOW).submitted == 1
    assert runtime.daily_budget(datetime(2026, 10, 14, 7, tzinfo=timezone.utc)).submitted == 0


def test_receipt_links_board_and_confirmation_urls_for_daily_capacity(data):
    saved = entry('https://board.example/job')
    saved['evidence'] = [{'url': 'https://employer.example/confirmation'}]
    (data / 'already_applied.json').write_text(json.dumps({'applications': [saved]}))
    (data / 'reports/employer-submitted-2026-10-13.json').write_text(json.dumps(saved))
    conn = init_db(data / 'applyagent.db')
    conn.execute('INSERT INTO jobs (url,application_url,title,company,apply_status,applied_at) '
                 'VALUES (?,?,?,?,?,?)', ('https://board.example/job', 'https://employer.example/apply',
                  'Junior Engineer', 'Employer', 'applied', '2026-10-13T09:00:00+00:00'))
    conn.commit()
    assert runtime.daily_budget(NOW).submitted == 1


def test_saved_daily_total_reserves_capacity_and_bad_state_stops(data):
    ledger = data / 'reports/applications-2026-10-13.json'
    ledger.write_text(json.dumps({'confirmed_submissions': 10}))
    assert runtime.daily_budget(NOW).remaining == 0
    ledger.write_text('invalid json')
    with pytest.raises(ValueError):
        runtime.daily_budget(NOW)


def test_application_lock_excludes_overlapping_runs_and_releases_on_error(data):
    with runtime.application_run_lock():
        with pytest.raises(RuntimeError, match='overlapping'):
            with runtime.application_run_lock():
                pytest.fail('second worker entered')
    with pytest.raises(ValueError):
        with runtime.application_run_lock():
            raise ValueError('worker failed')
    with runtime.application_run_lock():
        pass


def test_interrupted_worker_claim_blocks_restart_even_after_os_lock_released(data):
    conn = init_db(data / 'applyagent.db')
    conn.execute("INSERT INTO jobs (url, apply_status) VALUES ('job', 'in_progress')")
    conn.commit()
    with pytest.raises(RuntimeError, match='in-progress'):
        with runtime.application_run_lock():
            pytest.fail('new worker entered with unresolved old claim')


@pytest.mark.parametrize('reason', ['submission_uncertain', 'approval_required',
                                    'unknown_required_answer', 'captcha', 'site_blocked:403'])
def test_reset_and_direct_url_do_not_revive_protected_holds(data, monkeypatch, reason):
    from applyagent.apply import launcher
    conn = init_db(data / 'applyagent.db')
    conn.execute('INSERT INTO jobs (url,title,company,tailored_resume_path,apply_status,apply_error,apply_attempts) '
                 'VALUES (?,?,?,?,?,?,?)', (URL, 'Junior Engineer', 'Employer', '/resume.pdf', 'failed', reason, 99))
    conn.commit()
    monkeypatch.setattr(launcher, 'get_connection', lambda: conn)
    assert launcher.reset_failed() == 0
    assert launcher.acquire_job(target_url=URL + '/application') is None


def test_transient_reset_preserves_manual_jobs(data, monkeypatch):
    from applyagent.apply import launcher
    conn = init_db(data / 'applyagent.db')
    conn.execute("INSERT INTO jobs (url,apply_status,apply_error) VALUES ('temporary','failed','launcher_error')")
    conn.execute("INSERT INTO jobs (url,apply_status,apply_error) VALUES ('manual','manual','manual ATS')")
    conn.commit()
    monkeypatch.setattr(launcher, 'get_connection', lambda: conn)
    assert launcher.reset_failed() == 1
    assert conn.execute("SELECT apply_status FROM jobs WHERE url='manual'").fetchone()[0] == 'manual'


def test_direct_alias_cannot_retry_same_requisition(data, monkeypatch):
    from applyagent.apply import launcher
    conn = init_db(data / 'applyagent.db')
    conn.execute('INSERT INTO jobs (url,title,company,application_url,tailored_resume_path,apply_status,apply_attempts) '
                 'VALUES (?,?,?,?,?,?,?)',
                 ('https://board.example/held', 'Junior Engineer', 'Employer', URL, '/resume.pdf', 'failed', 99))
    conn.execute('INSERT INTO jobs (url,title,company,application_url,tailored_resume_path) VALUES (?,?,?,?,?)',
                 ('https://board.example/alias', 'Engineer I', 'Employer', URL + '/application', '/resume.pdf'))
    conn.commit()
    monkeypatch.setattr(launcher, 'get_connection', lambda: conn)
    assert launcher.acquire_job(target_url='https://board.example/alias') is None
