import httpx
import pytest

from applyagent import config, database, scheduled
from applyagent.discovery import employer_apis as api
from applyagent.discovery.quality import evaluate, experience_minima, has_early_experience, rank_jobs


@pytest.mark.parametrize('description', [
    '1–2 years of professional software engineering experience.',
    '2+ years of experience developing web applications.',
    '0 to 2 years of relevant experience.',
    '1-3 years building production software.',
])
def test_generic_swe_is_shortlisted_on_early_experience_evidence(description):
    job = {'title': 'Software Engineer II', 'location': 'New York, NY', 'company': 'Employer',
           'description': description, 'url': 'https://example.com/1'}
    assert has_early_experience(job)
    assert evaluate(job)[0] >= 60


def test_optional_years_and_benefit_durations_do_not_become_requirements():
    assert experience_minima('5 years of engineering experience preferred. 2 years of paid leave.') == []
    job = {'title': 'Software Engineer', 'location': 'NYC', 'company': 'Employer',
           'description': '4+ years of professional software development experience.'}
    assert evaluate(job)[0] == 0
    assert evaluate(dict(job, title='Senior Software Engineer', description='1-2 years of experience'))[0] == 0
    assert evaluate(dict(job, description='3 years of backend development. 1 year of experience with Node.'))[0] == 0


@pytest.mark.parametrize('description', [
    '3 to 5 years as a full-stack software engineer.',
    '3 years of professional iOS development.',
    '4 years of web engineering experience.',
])
def test_specialized_and_as_an_engineer_requirements_are_not_missed(description):
    assert evaluate({'title': 'Software Engineer', 'location': 'NYC',
                     'description': description})[0] == 0


def test_generic_swe_unknown_level_is_reviewable_with_description():
    job = {'title': 'Software Engineer', 'location': 'NYC', 'company': 'Employer',
           'description': 'Build maintainable services; collaborate with the team. ' * 10}
    assert evaluate(job)[0] >= 60
    assert 'verify' in ' '.join(evaluate(job)[2]).lower()
    assert evaluate(dict(job, location='Remote in Canada'))[0] == 0


def test_experience_evidence_ranks_ahead_of_founding_title():
    role = {'title': 'Software Engineer', 'location': 'NYC', 'company': 'Acme',
            'description': '1-2 years of experience', 'url': 'https://example.com/early'}
    assert rank_jobs([dict(role, title='Founding Engineer', url='https://example.com/founder'), role])[0]['url'] == role['url']


def test_shortlist_is_not_dominated_by_one_employer():
    base = {'title': 'Software Engineer', 'location': 'NYC', 'company': 'Large Employer',
            'description': '1-2 years of experience'}
    jobs = [dict(base, url=f'https://example.com/{i}') for i in range(10)]
    jobs.append(dict(base, company='Another Employer', url='https://example.com/other'))
    ranked = rank_jobs(jobs, max_per_company=3)
    assert len(ranked) == 4 and any(job['company'] == 'Another Employer' for job in ranked)


def test_greenhouse_decodes_description_and_skips_prospect_pool():
    source = {'provider': 'greenhouse', 'board': 'example', 'name': 'Example'}
    row = {'id': 1, 'title': 'Software Engineer', 'location': {'name': 'New York, NY'},
           'absolute_url': 'https://job-boards.greenhouse.io/example/jobs/1',
           'content': '&lt;p&gt;1-2 years of software experience&lt;/p&gt;'}
    jobs = api.parse_postings({'jobs': [row, dict(row, internal_job_id=None)]}, source)
    assert len(jobs) == 1
    assert jobs[0]['full_description'] == '1-2 years of software experience'


def test_lever_preserves_multiple_locations_requirements_and_application_url():
    source = {'provider': 'lever', 'board': 'example', 'name': 'Example'}
    payload = [{'text': 'Backend Engineer', 'categories': {'allLocations': ['San Francisco, CA', 'New York, NY']},
                'workplaceType': 'hybrid', 'description': '<p>Build services.</p>',
                'lists': [{'text': 'Requirements', 'content': '<li>1-2 years of experience</li>'}],
                'hostedUrl': 'https://jobs.lever.co/example/1', 'applyUrl': 'https://jobs.lever.co/example/1/apply'}]
    job = api.parse_postings(payload, source)[0]
    assert 'New York' in job['location'] and 'Requirements' in job['description']
    assert job['application_url'].endswith('/apply')
    assert evaluate(job)[0] >= 60


def test_ashby_preserves_remote_country_and_hides_unlisted_postings():
    source = {'provider': 'ashby', 'board': 'example', 'name': 'Example'}
    row = {'title': 'Software Engineer', 'location': 'United States', 'isRemote': True,
           'descriptionPlain': '1-2 years of experience', 'jobUrl': 'https://jobs.ashbyhq.com/example/1',
           'applyUrl': 'https://jobs.ashbyhq.com/example/1/application',
           'compensation': {'scrapeableCompensationSalarySummary': '$90K–$130K'}}
    jobs = api.parse_postings({'jobs': [row, dict(row, isListed=False)]}, source)
    assert len(jobs) == 1 and jobs[0]['location'] == 'Remote; United States'
    assert jobs[0]['salary'] == '$90K–$130K'
    hybrid = api.parse_postings({'jobs': [dict(row, workplaceType='Hybrid', location='San Francisco, CA')]}, source)[0]
    assert 'Remote' not in hybrid['location'] and evaluate(hybrid)[0] == 0


def test_new_york_city_full_state_name_and_mixed_international_offices():
    role = {'title': 'Software Engineer', 'company': 'Employer', 'description': '1-2 years of experience'}
    assert evaluate(dict(role, location='New York, New York'))[0] >= 60
    assert evaluate(dict(role, location='London; New York, NY'))[0] >= 60


def test_collector_is_get_only_keeps_healthy_source_and_skips_browser_enrichment(monkeypatch, tmp_path):
    conn = database.init_db(tmp_path / 'jobs.db')
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(api, 'get_connection', lambda: conn)
    boards = [{'name': 'Blocked', 'board': 'blocked', 'provider': 'greenhouse'},
              {'name': 'Healthy', 'board': 'healthy', 'provider': 'ashby'}]
    requests = []
    def respond(request):
        requests.append(request)
        if 'blocked' in request.url.path:
            return httpx.Response(403)
        return httpx.Response(200, json={'jobs': [{'title': 'Software Engineer', 'location': 'NYC',
            'descriptionPlain': '1-2 years of engineering experience',
            'jobUrl': 'https://jobs.ashbyhq.com/healthy/12345678-1234-1234-1234-123456789012',
            'applyUrl': 'https://jobs.ashbyhq.com/healthy/12345678-1234-1234-1234-123456789012/application'}]})
    real_client = httpx.Client
    monkeypatch.setattr(api.httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    cfg = {'employer_boards': boards}
    first = api.run_employer_apis(cfg)
    assert first['new'] == 1 and first['errors'] == 1
    saved = dict(conn.execute('SELECT * FROM jobs').fetchone())
    assert saved['detail_scraped_at'] and saved['application_url'] and saved['full_description']
    second = api.run_employer_apis(cfg)
    assert second['new'] == 0 and second['sources'][0]['status'] == 'backoff'
    assert all(request.method == 'GET' and not request.content for request in requests)


def test_employer_sources_validate_without_jobspy_queries():
    scheduled.validate_searches({'discovery_sources': ['employer_api'], 'employer_boards': [
        {'provider': 'lever', 'board': 'example', 'name': 'Example'}]})
    with pytest.raises(ValueError):
        api.endpoint({'provider': 'lever', 'board': '../private'})


def test_ordinary_discovery_uses_same_configured_sources_and_continues_after_error(monkeypatch):
    from unittest.mock import Mock

    from applyagent import pipeline
    from applyagent.discovery import public_sources
    cfg = {'discovery_sources': ['employer_api', 'public']}
    monkeypatch.setattr(config, 'load_search_config', lambda: cfg)
    monkeypatch.setattr(api, 'run_employer_apis', Mock(side_effect=RuntimeError('temporary source failure')))
    public = Mock(return_value={'new': 1, 'errors': 0})
    monkeypatch.setattr(public_sources, 'run_public_sources', public)
    results = pipeline._run_discover()
    assert results['employer_api']['errors'] == 1
    assert results['public']['new'] == 1
    public.assert_called_once_with(cfg)


def test_lever_view_and_apply_urls_have_one_identity():
    from applyagent.applications import application_identity
    url = 'https://jobs.lever.co/employer/12345678-1234-1234-1234-123456789012'
    assert application_identity(url) == application_identity(url + '/apply?lever-source=Indeed')


def test_budgeted_employer_boards_rotate_between_runs(monkeypatch, tmp_path):
    conn = database.init_db(tmp_path / 'jobs.db')
    monkeypatch.setattr(config, 'APP_DIR', tmp_path)
    monkeypatch.setattr(api, 'get_connection', lambda: conn)
    requests = []
    def respond(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={'jobs': []})
    real_client = httpx.Client
    monkeypatch.setattr(api.httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    cfg = {'employer_api_settings': {'max_boards': 1}, 'employer_boards': [
        {'name': name, 'board': name, 'provider': 'ashby'} for name in ['first', 'second', 'third']]}
    for unused in range(4):
        assert api.run_employer_apis(cfg)['errors'] == 0
    assert requests == ['/posting-api/job-board/' + name for name in ['first', 'second', 'third', 'first']]
