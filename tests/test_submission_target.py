from unittest.mock import Mock
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from applyagent.apply import launcher
from applyagent.apply.runtime import DailyBudget

DAY = datetime.now(timezone.utc).astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat()


def prepare_worker(monkeypatch):
    launcher._stop_event.clear()
    monkeypatch.setattr(launcher, 'launch_chrome', lambda *a, **kw: object())
    monkeypatch.setattr(launcher, 'cleanup_worker', Mock())
    monkeypatch.setattr(launcher, 'update_state', Mock())
    monkeypatch.setattr(launcher, 'add_event', Mock())
    monkeypatch.setattr(launcher, 'daily_budget', lambda: DailyBudget(DAY, 10, 0))


def test_target_counts_confirmed_successes_and_replenishes(monkeypatch):
    prepare_worker(monkeypatch)
    jobs = iter([{'url': 'a', 'title': 'A'}, None, {'url': 'b', 'title': 'B'}, {'url': 'c', 'title': 'C'}])
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kw: next(jobs))
    monkeypatch.setattr(launcher, 'launch_chrome', lambda *a, **kw: object())
    monkeypatch.setattr(launcher, 'cleanup_worker', Mock())
    monkeypatch.setattr(launcher, 'update_state', Mock())
    monkeypatch.setattr(launcher, 'add_event', Mock())
    outcomes = iter([('failed:temporary', 1), ('applied', 100), ('applied', 100)])
    monkeypatch.setattr(launcher, 'run_job', lambda *a, **kw: next(outcomes))
    marked = Mock()
    monkeypatch.setattr(launcher, 'mark_result', marked)
    refill = Mock(return_value=2)
    monkeypatch.setattr('applyagent.apply.refill.replenish', refill)
    assert launcher.worker_loop(limit=1, until_submitted=2) == (2, 1)
    refill.assert_called_once()
    assert marked.call_args_list[0].kwargs['permanent'] is True
    assert len(marked.call_args_list) == 3


def test_normal_run_replenishes_empty_queue_without_submission_target(monkeypatch):
    prepare_worker(monkeypatch)
    jobs = iter([None, {'url': 'new', 'title': 'New Grad Engineer'}])
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kw: next(jobs))
    monkeypatch.setattr(launcher, 'run_job', lambda *a, **kw: ('applied', 100))
    monkeypatch.setattr(launcher, 'mark_result', Mock())
    refill = Mock(return_value=1)
    monkeypatch.setattr('applyagent.apply.refill.replenish', refill)
    assert launcher.worker_loop(limit=1) == (1, 0)
    assert refill.call_args.kwargs['min_score'] == 7
    assert refill.call_args.kwargs['discover'] is True
    assert refill.call_args.kwargs['max_candidates'] == 12
    assert refill.call_args.kwargs['deadline'] > 0


def test_unreachable_submission_target_stops_without_polling_forever(monkeypatch):
    prepare_worker(monkeypatch)
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kw: None)
    refill = Mock(return_value=0)
    monkeypatch.setattr('applyagent.apply.refill.replenish', refill)
    assert launcher.worker_loop(until_submitted=10) == (0, 0)
    refill.assert_called_once()


def test_refill_failure_is_reported_as_error_not_an_empty_queue(monkeypatch):
    import pytest
    prepare_worker(monkeypatch)
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kwargs: None)
    monkeypatch.setattr('applyagent.apply.refill.replenish', Mock(side_effect=RuntimeError('Scoring unavailable')))
    with pytest.raises(RuntimeError, match='Application preparation failed'):
        launcher.worker_loop(limit=1)


def test_daily_cap_counts_previous_runs_and_stops_before_another_browser(monkeypatch):
    prepare_worker(monkeypatch)
    monkeypatch.setattr(launcher, 'daily_budget', lambda: DailyBudget(DAY, 10, 9))
    acquire = Mock(return_value={'url': 'new', 'title': 'New Grad Engineer'})
    monkeypatch.setattr(launcher, 'acquire_job', acquire)
    monkeypatch.setattr(launcher, 'run_job', lambda *a, **kw: ('applied', 100))
    monkeypatch.setattr(launcher, 'mark_result', Mock())
    assert launcher.worker_loop(until_submitted=10) == (1, 0)
    acquire.assert_called_once()


def test_cap_already_reached_never_claims_a_job(monkeypatch):
    prepare_worker(monkeypatch)
    monkeypatch.setattr(launcher, 'daily_budget', lambda: DailyBudget(DAY, 10, 10))
    acquire = Mock()
    monkeypatch.setattr(launcher, 'acquire_job', acquire)
    assert launcher.worker_loop(limit=5) == (0, 0)
    acquire.assert_not_called()


def test_no_refill_and_missing_specific_target_stop_without_discovery(monkeypatch):
    prepare_worker(monkeypatch)
    monkeypatch.setattr(launcher, 'acquire_job', lambda **kw: None)
    refill = Mock()
    monkeypatch.setattr('applyagent.apply.refill.replenish', refill)
    assert launcher.worker_loop(until_submitted=10, refill=False) == (0, 0)
    assert launcher.worker_loop(limit=0, target_url='https://example.com/held') == (0, 0)
    refill.assert_not_called()


def test_target_rejects_modes_that_cannot_reach_submission_count():
    import pytest
    with pytest.raises(ValueError, match='live queue'):
        launcher.worker_loop(until_submitted=10, dry_run=True)
    with pytest.raises(ValueError, match='live queue'):
        launcher.worker_loop(until_submitted=10, target_url='https://example.com/job')


def test_attempt_budget_counts_skips_without_opening_another_browser(monkeypatch):
    from applyagent import config
    prepare_worker(monkeypatch)
    monkeypatch.setattr(config, 'load_profile', lambda: {'application_workflow': {'max_attempts_per_run': 2}})
    acquire = Mock(return_value={'url': 'job', 'title': 'Junior Engineer'})
    monkeypatch.setattr(launcher, 'acquire_job', acquire)
    monkeypatch.setattr(launcher, 'run_job', lambda *args, **kwargs: ('skipped', 1))
    monkeypatch.setattr(launcher, 'mark_skipped', Mock())
    assert launcher.worker_loop(until_submitted=10) == (0, 0)
    assert acquire.call_count == 2


def test_expired_run_budget_stops_before_claiming_a_job(monkeypatch):
    from applyagent import config
    prepare_worker(monkeypatch)
    monkeypatch.setattr(config, 'load_profile', lambda: {'application_workflow': {'max_run_seconds': 1}})
    monkeypatch.setattr(launcher.time, 'monotonic', Mock(side_effect=[0, 2]))
    acquire = Mock()
    monkeypatch.setattr(launcher, 'acquire_job', acquire)
    assert launcher.worker_loop(until_submitted=10) == (0, 0)
    acquire.assert_not_called()


def test_refill_discovers_after_empty_description_failure(monkeypatch, tmp_path):
    from applyagent import config
    from applyagent.apply import refill
    from applyagent.database import init_db

    conn = init_db(tmp_path / 'jobs.db')
    conn.execute(
        "INSERT INTO jobs (url, title, full_description, application_url, detail_scraped_at) "
        "VALUES ('https://example.com/closed', 'Junior Software Engineer', '', "
        "'https://example.com/apply', '2026-10-11')")
    conn.commit()
    resume = tmp_path / 'resume.txt'
    resume.write_text('Candidate resume')
    monkeypatch.setattr(refill, 'get_connection', lambda: conn)
    monkeypatch.setattr(refill, 'filter_applications', lambda jobs: jobs)
    monkeypatch.setattr(refill, 'evaluate', lambda job: (100, []))
    monkeypatch.setattr(config, 'RESUME_PATH', resume)
    monkeypatch.setattr(config, 'load_profile', dict)
    discovery = Mock()
    scrape = Mock()
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    monkeypatch.setattr('applyagent.dedup.run_dedup', Mock())
    monkeypatch.setattr('applyagent.enrichment.detail.scrape_site_batch', scrape)
    assert refill.replenish() == 0
    discovery.assert_called_once()
    scrape.assert_not_called()


def test_enrichment_does_not_send_requisition_to_talent_community():
    from applyagent.enrichment.detail import application_target

    job = 'https://jobs.ashbyhq.com/miter/f4567649-a0eb-445d-bd00-a0673a26ec6d'
    assert application_target(job, '/miter/form/talent-community-form') == job
    assert application_target(job, job + '/application') == job + '/application'
    assert application_target('https://example.com/job/123', '/apply/123') == 'https://example.com/apply/123'
