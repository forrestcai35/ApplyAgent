from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from applyagent import cli, config
from applyagent.apply import backend, launcher
from applyagent.database import init_db


@pytest.fixture
def empty_queue(monkeypatch, tmp_path):
    conn = init_db(tmp_path / 'jobs.db')
    profile = tmp_path / 'profile.json'
    profile.write_text('{}')
    resume = tmp_path / 'resume.pdf'
    resume.write_bytes(b'test resume')
    monkeypatch.setattr(cli, '_bootstrap', Mock())
    monkeypatch.setattr(config, 'PROFILE_PATH', profile)
    monkeypatch.setattr(config, 'RESUME_PDF_PATH', resume)
    monkeypatch.setattr(config, 'get_chrome_path', lambda: '/test/chrome')
    monkeypatch.setattr(backend, 'selected_backend', lambda _: 'codex')
    monkeypatch.setattr(backend, 'executable', lambda _: '/test/codex')
    monkeypatch.setattr('applyagent.database.get_connection', lambda: conn)
    run = Mock()
    monkeypatch.setattr(launcher, 'main', run)
    return run


def test_cli_empty_queue_reaches_automatic_replenishment(empty_queue):
    result = CliRunner().invoke(cli.app, ['apply', '--limit', '1'])
    assert result.exit_code == 0, result.output
    assert empty_queue.call_args.kwargs['refill'] is True
    assert empty_queue.call_args.kwargs['refill_rounds'] == 3


def test_cli_opt_out_keeps_existing_queue_requirement(empty_queue):
    result = CliRunner().invoke(cli.app, ['apply', '--limit', '1', '--no-refill'])
    assert result.exit_code == 1
    assert 'No jobs ready' in result.output
    empty_queue.assert_not_called()


def test_cli_reports_overlap_without_starting_another_worker(empty_queue):
    empty_queue.side_effect = RuntimeError('Another application run is active')
    result = CliRunner().invoke(cli.app, ['apply', '--limit', '1'])
    assert result.exit_code == 1
    assert 'Another application run is active' in result.output
