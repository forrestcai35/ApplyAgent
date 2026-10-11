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
