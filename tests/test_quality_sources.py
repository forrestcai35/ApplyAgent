import json

import pytest

from applyagent.discovery.github_repos import _extract_apply_url, _parse_html_table
from applyagent.discovery.public_sources import parse_feed, parse_page
from applyagent.discovery.quality import canonical_url, evaluate, rank_jobs


def job(title="Junior Software Engineer", location="NYC", **extra):
    return {"title": title, "location": location, "url": "https://example.com/jobs/1", "company": "Acme", **extra}


@pytest.mark.parametrize("title,location", [
    ("Senior Software Engineer", "NYC"), ("Junior Mechanical Engineer", "NYC"),
    ("Junior Software Engineer", "Albany, NY"), ("Junior Software Engineer", "New York City Metropolitan Area"),
    ("Junior Software Engineer", "Boston (not remote)"), ("Junior Software Engineer", None),
])
def test_quality_rejects_wrong_role_or_location(title, location):
    assert evaluate(job(title, location))[0] == 0


def test_explicit_fit_wins_regardless_of_platform():
    roles = [job("Software Engineer", site="linkedin"),
             job("Founding Engineer", "Remote", site="anything", url="https://example.com/2")]
    result = rank_jobs(roles, min_score=60)
    assert len(result) == 1
    assert result[0]["title"] == "Founding Engineer"
    assert "Confirm required experience for founding role" in result[0]["review_needed"]


def test_junior_candidate_is_prioritized_over_higher_scored_founding_role():
    roles = [job('Founding Engineer', description='Full description ' * 30),
             job('Software Engineer I', url='https://example.com/junior')]
    assert rank_jobs(roles)[0]['title'] == 'Software Engineer I'


def test_tracking_dedup_preserves_job_ids():
    assert canonical_url("https://example.com/job?gh_jid=12&utm_source=list#apply") == "https://example.com/job?gh_jid=12"
    assert len(rank_jobs([job(), job(url="https://example.com/jobs/1?utm_source=list")])) == 1


def test_github_inherited_company_and_closed_rows():
    text = '''<table><tr><td>Acme</td><td>Software Engineer 1</td><td>NYC</td><td><a href="https://example.com/1">Apply</a></td><td>3d</td></tr>
    <tr><td>↳</td><td>Junior Developer</td><td>Remote</td><td><a href="https://example.com/2">Apply</a></td><td>5d</td></tr>
    <tr><td>Other</td><td>Junior Developer</td><td>NYC</td><td>🔒</td><td>1d</td></tr></table>'''
    rows = _parse_html_table(text, {"company": 0, "title": 1, "location": 2, "apply": 3, "age": 4})
    assert [row["company"] for row in rows] == ["Acme", "Acme"]
    assert rows[1]["age_days"] == 5
    assert rows[1]["title"] == "Junior Developer"
    assert _extract_apply_url("[Apply](https://example.com/job)") == "https://example.com/job"


def test_rss_remote_region_and_expiration():
    text = '''<rss><channel><item><title>Acme: Junior Developer</title><region>USA only</region><link>https://example.com/1</link></item>
    <item><title>Expired</title><link>https://example.com/2</link><expires_at>2020-01-01</expires_at></item></channel></rss>'''
    rows = parse_feed(text, {"remote": True, "company_prefix": True})
    assert len(rows) == 1
    assert rows[0]["location"] == "Remote — USA only"
    assert rows[0]["company"] == "Acme"


def test_public_html_structured_jobs_and_expiry():
    data = {"@graph": [
        {"@type": "JobPosting", "title": "Junior Developer", "hiringOrganization": {"name": "Acme"},
         "jobLocation": {"address": {"addressLocality": "New York", "addressRegion": "NY"}}},
        {"@type": "JobPosting", "title": "Expired", "validThrough": "2020-01-01"},
    ]}
    rows = parse_page('<script type="application/ld+json">'+json.dumps(data)+'</script>', "https://example.com/job")
    assert len(rows) == 1
    assert rows[0]["location"] == "New York, NY"
    assert parse_page('<form>Sign in</form>', "https://example.com/login") == []


@pytest.mark.parametrize("role", [
    job(location="Remote in Canada"), job("Founding Engineer (PhD)"),
    job("Founding Customer Success Engineer"), job("Founding Rover Systems Engineer", "Remote"),
    job(description="A PhD in progress or already completed is required for this role."),
    job("Founding ML Engineer", description="5+ years in applied computer vision"),
])
def test_resume_search_exclusions(role):
    assert evaluate(role)[0] == 0


def test_optional_phd_and_us_canada_are_not_hard_rejections():
    assert evaluate(job(location="Remote USA or Canada", description="PhD preferred but not required."))[0] > 0
