import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from applyagent import config, review
from applyagent.discovery import jobspy
from applyagent.discovery.backoff import BoardBackoff


@pytest.fixture
def private_state(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(config, 'load_profile', lambda: {'application_workflow': {'skip_required_fields': ['gpa', 'transcript']}})
    return tmp_path


def test_exact_hold_suppresses_tracking_alias_but_not_distinct_role(private_state):
    url = 'https://jobs.ashbyhq.com/acme/12345678-1234-1234-1234-123456789012/application'
    review.remember({'status': 'local_package_held', 'company': 'Acme',
                     'employer_application_url': url, 'reason': 'Missing exact authorization'})
    assert review.deferred_reason({'application_url': url + '?utm_source=board'})
    assert review.deferred_reason({'company': 'Acme', 'url': 'https://example.com/different'}) is None
    # Even far in the future, an authorization hold must not grant permission.
    assert review.deferred_reason({'url': url}, now=datetime.now(UTC) + timedelta(days=30))


def test_closed_posting_can_be_rechecked_without_refresh_extending_hold(private_state):
    checked = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    entry = {'status': 'closed', 'url': 'https://example.com/closed', 'reason': 'Job not found', 'checked_at': checked}
    reports = private_state / 'reports'; reports.mkdir()
    (reports / 'applications-2026-10-14.json').write_text(json.dumps({'review': [entry]}))
    review.refresh_from_reports(); review.refresh_from_reports()
    assert review.deferred_reason({'url': entry['url']}) is None
    assert review._load()['entries'][entry['url']]['checked_at'] == checked


def test_runtime_limits_do_not_release_fact_holds(private_state, monkeypatch):
    profile = {'personal': {'city': 'San Jose'}, 'application_workflow': {'max_run_seconds': 900}}
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    review.remember({'url': 'https://example.com/held', 'status': 'held', 'reason': 'Remote eligibility unverified'})
    profile['application_workflow']['max_run_seconds'] = 180
    assert review.deferred_reason({'url': 'https://example.com/held'})
    profile['personal']['city'] = 'New York'
    assert review.deferred_reason({'url': 'https://example.com/held'}) is None


def test_remote_and_interview_preferences_do_not_release_experience_holds(private_state, monkeypatch):
    profile = {'personal': {'city': 'San Jose'},
               'experience': {'years_of_experience_total': 1}, 'preferences': {},
               'availability': {'can_relocate': True}}
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    review.remember({'url': 'https://example.com/experience',
                     'status': 'excluded_documented_experience', 'reason': 'Requires two years'})
    profile['preferences']['remote_application_policy'] = 'us_remote_unless_explicitly_excluded'
    profile['availability']['interview_process_timing'] = 'As early as possible'
    profile['personal']['city'] = 'New York'
    assert review.deferred_reason({'url': 'https://example.com/experience'})
    profile['experience']['years_of_experience_total'] = 2
    assert review.deferred_reason({'url': 'https://example.com/experience'}) is None


def test_equal_timestamp_package_enriches_required_fields(private_state):
    reports = private_state / 'reports'; reports.mkdir()
    entry = {'status': 'local_package_held', 'url': 'https://example.com/job',
             'checked_at': datetime.now(UTC).isoformat(), 'reason': 'Authorization absent'}
    (reports / 'applications-2026-10-14.json').write_text(json.dumps({'review': [entry]}))
    entry['mandatory_unknowns'] = ['Required GPA category', 'Required undergraduate transcript']
    (reports / 'application-packages-2026-10-14.json').write_text(json.dumps({'packages': [entry]}))
    queue = review.queue_summary(refresh=True)
    assert queue['excluded_by_preference'] == 1
    assert queue['items'][0]['authorization'].startswith('Not established')


def test_unrelated_ledger_update_does_not_erase_mandatory_field_hold(private_state):
    reports = private_state / 'reports'; reports.mkdir()
    entry = {'status': 'excluded', 'url': 'https://example.com/job',
             'checked_at': '2026-10-14T10:00:00+00:00', 'reason': 'Mandatory GPA',
             'mandatory_unknowns': ['Required GPA category']}
    review.remember(entry)
    (reports / 'applications-2026-10-14.json').write_text(json.dumps({
        'updated_at': '2026-10-14T20:00:00+00:00',
        'review': [{'status': 'held', 'url': entry['url'], 'reason': 'Old summary'}]}))
    queue = review.queue_summary(refresh=True)
    assert queue['excluded_by_preference'] == 1
    assert queue['items'][0]['checked_at'] == entry['checked_at']


def test_cross_post_alias_does_not_create_duplicate_cached_item(private_state):
    reports = private_state / 'reports'; reports.mkdir()
    checked = datetime.now(UTC).isoformat()
    entry = {'status': 'held', 'source_url': 'https://example.com/source',
             'employer_application_url': 'https://example.com/employer',
             'checked_at': checked, 'reason': 'Authorization absent'}
    document = {'review': [entry], 'discovery': {'new_jobs': [
        {'url': entry['source_url'], 'application_url': 'https://example.com/shortlink'}]}}
    (reports / 'applications-2026-10-14.json').write_text(json.dumps(document))
    (reports / 'application-packages-2026-10-14.json').write_text(json.dumps({'packages': [
        dict(entry, mandatory_unknowns=['Required GPA category'])]}))
    assert review.queue_summary(refresh=True)['held_requisitions'] == 1
    assert review.queue_summary(refresh=True)['held_requisitions'] == 1
    assert review.deferred_reason({'url': 'https://example.com/new-source',
                                   'application_url': 'https://example.com/shortlink'})
    review.remember(dict(entry, reason='New exact authorization hold',
                         checked_at=datetime.now(UTC).isoformat()))
    assert review.queue_summary()['held_requisitions'] == 1
    assert review.deferred_reason({'url': 'https://example.com/shortlink'}) == 'New exact authorization hold'


def test_report_claiming_authorization_cannot_create_positive_cache(private_state):
    assert review.remember({'status': 'authorized', 'url': 'https://example.com/job'}) is None
    assert not review._load()['entries']


def test_skip_policy_applies_only_to_known_required_fields(private_state):
    assert review.excluded_required_fields(['What is your GPA?', 'Upload undergraduate transcript'])
    assert review.excluded_required_fields(['Resume', 'Expected graduation month']) == []
    # Caller supplies only required controls: optional GPA must not be included.
    assert review.excluded_required_fields([]) == []
    review.remember({'status': 'excluded', 'url': 'https://example.com/gpa',
                     'reason': 'Preference', 'mandatory_unknowns': ['Required GPA category']})
    assert review.deferred_reason({'url': 'https://example.com/gpa'},
                                  now=datetime.now(UTC) + timedelta(days=30))


def test_rate_limit_is_respected_across_queries_and_runs(private_state):
    backoff = BoardBackoff()
    backoff.handle(logging.LogRecord('JobSpy:LinkedIn', logging.ERROR, '', 0, 'HTTP 429', (), None))
    assert not backoff.allows('linkedin')
    assert backoff.allows('indeed')
    assert not BoardBackoff().allows('linkedin')


def test_parser_failure_stops_board_and_dns_stops_remaining_queries(private_state):
    backoff = BoardBackoff()
    backoff.handle(logging.LogRecord('JobSpy:ZipRecruiter', logging.ERROR, '', 0, 'no jobKeysMap in the page', (), None))
    assert not backoff.allows('zip_recruiter')
    backoff.failed('indeed', 'Failed to resolve apis.indeed.com')
    assert not backoff.allows('indeed') and not backoff.allows('github')
    assert BoardBackoff().allows('indeed')  # DNS sandbox failures are run-local.


def test_restricted_exception_does_not_retry_or_sleep(private_state, monkeypatch):
    scrape = Mock(side_effect=RuntimeError('HTTP 403 connection refused'))
    sleep = Mock()
    monkeypatch.setattr(jobspy, 'scrape_jobs', scrape)
    monkeypatch.setattr(jobspy.time, 'sleep', sleep)
    with pytest.raises(RuntimeError):
        jobspy._scrape_with_retry({})
    scrape.assert_called_once(); sleep.assert_not_called()


def test_crawl_removes_error_handlers_even_on_failure(private_state, monkeypatch):
    monkeypatch.setattr(jobspy, 'init_db', Mock())
    monkeypatch.setattr(jobspy, '_crawl_searches', Mock(side_effect=RuntimeError('failed')))
    with pytest.raises(RuntimeError):
        jobspy._full_crawl({})
    assert not any(isinstance(handler, BoardBackoff) for handler in logging.getLogger('JobSpy:LinkedIn').handlers)


def test_logged_429_is_not_queried_again_in_same_crawl(private_state, monkeypatch):
    import pandas as pd

    from applyagent.database import init_db
    conn = init_db(private_state / 'jobs.db')
    monkeypatch.setattr(jobspy, 'init_db', lambda: conn)
    monkeypatch.setattr(jobspy, 'get_connection', lambda: conn)
    calls = []
    def scrape(kwargs, **unused):
        calls.append(kwargs['site_name'][0])
        if kwargs['site_name'] == ['linkedin']:
            logging.getLogger('JobSpy:LinkedIn').error('HTTP 429')
        return pd.DataFrame()
    monkeypatch.setattr(jobspy, '_scrape_with_retry', scrape)
    result = jobspy._full_crawl({'queries': [{'query': 'junior'}, {'query': 'graduate'}],
                                'locations': [{'location': 'New York, NY'}]}, sites=['linkedin', 'indeed'])
    assert calls == ['linkedin', 'indeed', 'indeed']
    assert result['backoff_skips'] == 1
    assert result['backoff_failures'] == 1
