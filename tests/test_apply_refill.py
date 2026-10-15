from unittest.mock import Mock

import pytest

from applyagent import config
from applyagent.apply import refill
from applyagent.database import init_db


@pytest.fixture
def queue(monkeypatch, tmp_path):
    conn = init_db(tmp_path / 'jobs.db')
    resume = tmp_path / 'resume.txt'
    resume.write_text('Documented candidate experience')
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(config, 'RESUME_PATH', resume)
    monkeypatch.setattr(config, 'RESUME_PDF_PATH', tmp_path / 'resume.pdf')
    monkeypatch.setattr(config, 'load_profile', dict)
    monkeypatch.setattr(config, 'load_search_config', dict)
    monkeypatch.setattr(refill, 'get_connection', lambda: conn)
    monkeypatch.setattr('applyagent.dedup.run_dedup', Mock())
    return conn


def insert(conn, url, title='Junior Software Engineer'):
    conn.execute('INSERT INTO jobs (url,title,company,site,location,full_description,application_url) '
                 'VALUES (?,?,?,?,?,?,?)',
                 (url, title, 'Employer', 'employer', 'New York, NY',
                  'An employer description with documented role requirements. ' * 10, url + '/application'))
    conn.commit()


def test_rejected_first_batch_does_not_hide_next_suitable_candidate(queue, monkeypatch):
    for index in range(21):
        insert(queue, f'https://example.com/jobs/{index:03}')
    score = Mock(side_effect=lambda resume, job: {
        'score': 8 if job['url'].endswith('020') else 4, 'reasoning': 'Verified fit decision'})
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    discovery = Mock()
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    assert refill.replenish(batch_size=20) == 1
    assert score.call_count == 21
    discovery.assert_not_called()
    assert queue.execute("SELECT tailored_resume_path FROM jobs WHERE url LIKE '%020'").fetchone()[0]


def test_explicit_junior_roles_are_checked_before_founding_roles(queue, monkeypatch):
    insert(queue, 'https://example.com/a', 'Founding Engineer')
    insert(queue, 'https://example.com/z', 'Software Engineer I')
    score = Mock(return_value={'score': 8, 'reasoning': 'Fit'})
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    assert refill.replenish(batch_size=1) == 1
    assert score.call_args.args[1]['title'] == 'Software Engineer I'


def test_refill_discovers_once_after_existing_candidates_are_screened_out(queue, monkeypatch):
    insert(queue, 'https://example.com/rejected')
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', lambda *args: {'score': 4, 'reasoning': 'Mismatch'})
    discovery = Mock()
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    assert refill.replenish() == 0
    discovery.assert_called_once_with(timeout=600)


def test_model_failure_preserves_candidate_without_discovery_or_low_score(queue, monkeypatch):
    insert(queue, 'https://example.com/job')
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', lambda *args: {'score': 0, 'reasoning': 'Unavailable'})
    discovery = Mock()
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    with pytest.raises(RuntimeError, match='Scoring unavailable'):
        refill.replenish()
    assert queue.execute('SELECT fit_score FROM jobs').fetchone()[0] is None
    discovery.assert_not_called()


def test_replenishment_bound_and_no_discovery_are_honored(queue, monkeypatch):
    for index in range(4):
        insert(queue, f'https://example.com/{index}')
    score = Mock(return_value={'score': 4, 'reasoning': 'Mismatch'})
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    discovery = Mock()
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    assert refill.replenish(max_candidates=2, discover=False) == 0
    assert score.call_count == 2
    discovery.assert_not_called()


def test_expired_refill_deadline_never_scores_or_discovers(queue, monkeypatch):
    insert(queue, 'https://example.com/new')
    score = Mock()
    discovery = Mock()
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    monkeypatch.setattr('applyagent.scheduled.run_scheduled', discovery)
    assert refill.replenish(deadline=0) == 0
    score.assert_not_called(); discovery.assert_not_called()


def test_cached_exact_hold_never_scores_cross_post(queue, monkeypatch):
    from applyagent.review import remember
    insert(queue, 'https://example.com/held')
    remember({'status': 'held', 'url': 'https://example.com/held/application', 'reason': 'Authorization unavailable'})
    score = Mock()
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    assert refill.replenish(discover=False) == 0
    score.assert_not_called()


def test_refill_batch_includes_other_employer_before_large_employer_overflow(queue, monkeypatch):
    for index in range(9):
        insert(queue, f'https://example.com/a{index}')
    insert(queue, 'https://example.com/z')
    queue.execute("UPDATE jobs SET company='Other Employer' WHERE url='https://example.com/z'")
    queue.commit()
    score = Mock(return_value={'score': 8, 'reasoning': 'Fit'})
    monkeypatch.setattr('applyagent.scoring.scorer.score_job', score)
    assert refill.replenish(batch_size=12, discover=False) == 4
    assert any(call.args[1]['company'] == 'Other Employer' for call in score.call_args_list)
