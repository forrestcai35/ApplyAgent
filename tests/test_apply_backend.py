import json
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from applyagent.apply import backend


def test_selection_follows_text_backend_and_explicit_override(monkeypatch):
    monkeypatch.delenv('APPLYAGENT_APPLY_BACKEND', raising=False)
    monkeypatch.setenv('APPLYAGENT_LLM_BACKEND', 'codex')
    assert backend.selected_backend() == 'codex'
    assert backend.selected_backend('claude') == 'claude'
    monkeypatch.setenv('APPLYAGENT_APPLY_BACKEND', 'claude')
    assert backend.selected_backend() == 'claude'
    with pytest.raises(ValueError):
        backend.selected_backend('bad')


@pytest.mark.parametrize('output,expected', [
    ('RESULT:APPLIED', 'failed:submission_uncertain'),
    ('RESULT:APPLIED\nCONFIRMATION_URL: https://jobs.example/thanks\nCONFIRMATION_MESSAGE: Application received', 'applied'),
    ('RESULT:FAILED:unknown_required_answer', 'failed:unknown_required_answer'),
    ('RESULT:CAPTCHA', 'captcha'),
    ('RESULT:APPLIED\nRESULT:EXPIRED', 'failed:submission_uncertain'),
    ('The instructions say RESULT:APPLIED', 'failed:submission_uncertain'),
    ('', 'failed:submission_uncertain'),
])
def test_only_one_final_result_with_evidence(output, expected):
    assert backend.parse_result(output) == expected


def test_dry_run_never_counts_submission():
    assert backend.parse_result('RESULT:APPLIED', dry_run=True) == 'failed:dry_run_not_submitted'


def test_scoped_command_and_default_model(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, 'executable', lambda _: '/bin/codex')
    cmd = backend.codex_command(9555, tmp_path / 'answer')
    assert '--model' not in cmd
    assert '--dangerously-bypass-approvals-and-sandbox' not in cmd
    assert 'features.shell_tool=false' in cmd
    assert 'mcp_servers.playwright.command="npx"' in cmd
    assert '--cdp-endpoint=http://localhost:9555' in ' '.join(cmd)
    assert '--model' in backend.codex_command(9555, tmp_path / 'answer', 'chosen-model')


def test_run_uses_final_file_not_stream_and_reaps_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, 'executable', lambda _: '/bin/codex')
    from applyagent.apply import chrome  # noqa: F401
    proc = Mock(returncode=0, pid=123)
    proc.poll.return_value = 0
    def spawn(cmd, **kwargs):
        Path(cmd[cmd.index('--output-last-message') + 1]).write_text('RESULT:CAPTCHA')
        return proc
    monkeypatch.setattr(backend.subprocess, 'Popen', spawn)
    register, unregister = Mock(), Mock()
    args = ('prompt', 9555, tmp_path, tmp_path / 'run.log', None, False, 3, register, unregister)
    assert backend.run_codex(*args) == 'captcha'
    register.assert_called_once_with(proc)
    unregister.assert_called_once()
    kill = Mock()
    monkeypatch.setattr('applyagent.apply.chrome._kill_process_tree', kill)
    proc.communicate.side_effect = [subprocess.TimeoutExpired('codex', 3), None]
    assert backend.run_codex(*args) == 'failed:submission_uncertain'
    kill.assert_called_once_with(123)


def test_queue_skips_registry_match_and_selects_next(monkeypatch, tmp_path):
    from applyagent import config
    from applyagent.apply import launcher
    c = sqlite3.connect(':memory:')
    c.row_factory = sqlite3.Row
    c.execute('''CREATE TABLE jobs (url TEXT, title TEXT, company TEXT, site TEXT,
        application_url TEXT, tailored_resume_path TEXT, fit_score INTEGER,
        location TEXT, full_description TEXT, cover_letter_path TEXT,
        duplicate_of TEXT, apply_status TEXT, apply_attempts INTEGER,
        applied_at TEXT, agent_id TEXT, last_attempted_at TEXT)''')
    for company, url in [('Already', 'https://example.com/a'), ('Fresh', 'https://example.com/b')]:
        c.execute('INSERT INTO jobs (url,title,company,site,tailored_resume_path,fit_score) VALUES (?,?,?,?,?,?)',
                  (url, 'Engineer', company, 'employer', '/resume.pdf', 8))
    c.commit()
    (tmp_path / 'already_applied.json').write_text(json.dumps({'applications': [
        {'company': 'Already', 'role': 'Engineer', 'scope': '2026'}]}))
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(config, 'is_manual_ats', lambda _: False)
    monkeypatch.setattr(launcher, 'get_connection', lambda: c)
    monkeypatch.setattr(launcher, '_load_blocked', lambda: (set(), []))
    assert launcher.acquire_job()['company'] == 'Fresh'
    assert launcher.acquire_job(target_url='https://example.com/a') is None
    assert launcher._is_permanent_failure('failed:submission_uncertain')


def test_target_query_identifies_exact_requisition(monkeypatch):
    from applyagent.apply import launcher
    c = sqlite3.connect(':memory:')
    c.row_factory = sqlite3.Row
    c.execute('''CREATE TABLE jobs (url TEXT, title TEXT, company TEXT, site TEXT,
        application_url TEXT, tailored_resume_path TEXT, fit_score INTEGER,
        location TEXT, full_description TEXT, cover_letter_path TEXT,
        apply_status TEXT, applied_at TEXT, agent_id TEXT, last_attempted_at TEXT)''')
    for key in ('wrong', 'wanted'):
        c.execute('INSERT INTO jobs (url,title,company,tailored_resume_path) VALUES (?,?,?,?)',
                  (f'https://www.indeed.com/viewjob?jk={key}', key, key, '/resume.pdf'))
    c.commit()
    monkeypatch.setattr(launcher, 'get_connection', lambda: c)
    monkeypatch.setattr('applyagent.applications.filter_applications', lambda jobs: jobs)
    monkeypatch.setattr('applyagent.config.is_manual_ats', lambda _: False)
    assert launcher.acquire_job(target_url='https://www.indeed.com/viewjob?jk=wanted')['title'] == 'wanted'


def test_cleanup_does_not_kill_an_unowned_browser(monkeypatch):
    from applyagent.apply import chrome
    monkeypatch.setattr(chrome, '_chrome_procs', {})
    port_kill = Mock()
    monkeypatch.setattr(chrome, '_kill_on_port', port_kill)
    chrome.kill_all_chrome()
    chrome.cleanup_on_exit()
    port_kill.assert_not_called()
