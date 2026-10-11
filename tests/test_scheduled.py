import fcntl
import json
import subprocess
from unittest.mock import Mock

import pytest

from applyagent import config, database, scheduled
from applyagent.discovery import jobspy


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for key, value in {
        "APP_DIR": tmp_path, "LOG_DIR": tmp_path / "logs",
        "SEARCH_CONFIG_PATH": tmp_path / "searches.yaml",
    }.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(config, "ensure_dirs", lambda: config.LOG_DIR.mkdir(exist_ok=True))
    conn = database.init_db(tmp_path / "test.db")
    monkeypatch.setattr(database, "init_db", lambda path=None: conn)
    config.SEARCH_CONFIG_PATH.write_text('queries:\n  - query: engineer\nlocations:\n  - location: Remote\nboards: [linkedin]\n')
    return tmp_path


def test_missing_config_never_uses_example(configured):
    config.SEARCH_CONFIG_PATH.unlink()
    assert scheduled.run_scheduled() == 2


@pytest.mark.parametrize("cfg", [None, {}, {"queries": []}, {"queries": [{"query": ""}]}])
def test_invalid_config(cfg):
    with pytest.raises((ValueError, TypeError)):
        scheduled.validate_searches(cfg)


def test_overlap_skips_without_spawning(configured, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with (configured / "discovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert scheduled.run_scheduled() == 0
    spawn.assert_not_called()


def test_timeout_kills_worker_and_records_failure(configured, monkeypatch):
    process = Mock(pid=123)
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), -9]
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(scheduled.os, "killpg", kill)
    assert scheduled.run_scheduled(timeout=1) == 124
    assert kill.call_count == 1
    assert json.loads((configured / "reports/latest.json").read_text())["status"] == "timeout"


def test_worker_crash_is_reported(configured, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=Mock(wait=Mock(return_value=1))))
    assert scheduled.run_scheduled() == 1
    assert json.loads((configured / "reports/latest.json").read_text())["status"] == "error"


def test_worker_reports_only_new_jobs_and_partial_errors(configured, monkeypatch):
    conn = database.init_db(configured / "test.db")
    conn.execute("INSERT INTO jobs (url, title) VALUES ('https://example.org/old', 'Old')")
    conn.commit()
    monkeypatch.setattr(database, "init_db", lambda: conn)
    monkeypatch.setattr(config, "load_env", lambda: None)

    def discover(cfg):
        conn.execute("INSERT OR IGNORE INTO jobs (url, title) VALUES ('https://example.org/new', 'New')")
        conn.commit()
        return {"queries": 1, "errors": 1}

    monkeypatch.setattr(jobspy, "run_discovery", discover)
    report_path = configured / "result.json"
    assert scheduled._worker(report_path) == 1
    assert [j["title"] for j in json.loads(report_path.read_text())["new_jobs"]] == ["New"]
    scheduled._worker(report_path)
    assert json.loads(report_path.read_text())["new_jobs"] == []


def test_malformed_yaml_is_config_error(configured):
    config.SEARCH_CONFIG_PATH.write_text('queries: [')
    assert scheduled.run_scheduled() == 2


def test_logged_board_errors_fail_the_run(configured, monkeypatch):
    import logging

    conn = database.init_db(configured / "test.db")
    monkeypatch.setattr(database, "init_db", lambda: conn)
    monkeypatch.setattr(config, "load_env", lambda: None)
    logger = logging.getLogger("JobSpy:TestBoard")

    def discover(cfg):
        logger.error("HTTP 429")
        return {"queries": 1, "errors": 0}

    monkeypatch.setattr(jobspy, "run_discovery", discover)
    assert scheduled._worker(configured / "result.json") == 1
    assert not any(isinstance(h, scheduled._BoardErrors) for h in logger.handlers)


def test_interrupted_run_preserves_new_jobs_in_report(configured, monkeypatch):
    conn = database.init_db()

    def spawn(*args, **kwargs):
        conn.execute("INSERT INTO jobs (url, title) VALUES ('https://example.org/job', 'Engineer')")
        conn.commit()
        return Mock(wait=Mock(return_value=1))

    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert scheduled.run_scheduled() == 1
    report = json.loads((configured / "reports/latest.json").read_text())
    assert report["new_jobs"][0]["title"] == "Engineer"


def test_source_failure_does_not_stop_other_sources(configured, monkeypatch):
    from applyagent.discovery import github_repos, public_sources

    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_search_config", lambda: {"discovery_sources": ["github", "public"]})
    monkeypatch.setattr(github_repos, "run_github_repos", Mock(side_effect=RuntimeError("offline")))
    public = Mock(return_value={"sources": [{"source": "feed"}], "errors": 0})
    monkeypatch.setattr(public_sources, "run_public_sources", public)
    report = configured / "result.json"
    assert scheduled._worker(report) == 1
    public.assert_called_once()
    assert json.loads(report.read_text())["stats"]["sources"]["github"]["errors"] == 1
