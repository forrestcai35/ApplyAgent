import json

from applyagent.applications import filter_applications


def test_suppresses_role_but_preserves_distinct_and_historical_openings(tmp_path):
    path = tmp_path / "already_applied.json"
    path.write_text(json.dumps({"applications": [
        {"company": "Clay Labs", "role": "Early Career Software Engineer", "scope": "2026"},
        {"company": "Google", "role": None, "scope": "2026"},
        {"company": "Stripe", "role": "Software Engineer", "scope": "historical"},
    ]}))
    jobs = [
        {"company": "Clay", "title": "Early Career Software Engineer"},
        {"company": "Clay", "title": "Founding Engineer"},
        {"company": "Google", "title": "Software Engineer"},
        {"company": "Stripe", "title": "Software Engineer"},
        {"company": "New Company", "title": "Software Engineer"},
    ]
    result = filter_applications(jobs, path)
    assert len(result) == 4
    assert all("application_history" in j for j in result[:3])
    assert "application_history" not in result[3]
    assert "application_history" not in jobs[1]


def test_missing_registry_is_optional(tmp_path):
    jobs = [{"company": "Clay", "title": "Engineer"}]
    assert filter_applications(jobs, tmp_path / "missing.json") == jobs


def test_suppresses_renamed_employer_using_submission_evidence(tmp_path):
    path = tmp_path / 'already_applied.json'
    job_id = '67d0c7d0-fddb-4b8d-aed8-0647337a988e'
    path.write_text(json.dumps({'applications': [{
        'company': 'Old Name', 'role': 'Software Engineer', 'submission_confirmed': True,
        'evidence': [{'url': f'https://jobs.ashbyhq.com/old-name/{job_id}/application'}],
    }]}))
    jobs = [
        {'company': 'New Name', 'title': 'New Grad Engineer',
         'url': f'https://jobs.ashbyhq.com/new-name/{job_id}/'},
        {'company': 'New Name', 'title': 'Distinct Engineer',
         'url': 'https://jobs.ashbyhq.com/new-name/927c74a5-1b40-490d-b3be-a29baec4db6f'},
    ]
    assert filter_applications(jobs, path) == jobs[1:]
def test_greenhouse_board_and_embedded_forms_share_exact_identity():
    from applyagent.applications import application_identity
    assert application_identity('https://boards.greenhouse.io/acme/jobs/123?utm_source=board') == \
        application_identity('https://job-boards.greenhouse.io/acme/jobs/123') == \
        application_identity('https://job-boards.greenhouse.io/embed/job_app?for=acme&token=123')
    assert application_identity('https://job-boards.greenhouse.io/acme/jobs/124') != \
        application_identity('https://job-boards.greenhouse.io/acme/jobs/123')
