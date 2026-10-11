"""Regression coverage for application answer context and prompt assembly."""
import pytest

from applyagent.apply import prompt


@pytest.fixture
def applicant():
    return {
        'personal': {'full_name': 'Test Candidate', 'email': 'test@example.com',
                     'phone': '5550100', 'city': 'San Jose'},
        'work_authorization': {'legally_authorized_to_work': True,
                               'require_sponsorship': False,
                               'future_sponsorship_required': False,
                               'citizenship': 'United States'},
        'experience': {'years_of_experience_total': 0, 'school': 'Example University',
                       'education_level': 'BS Computer Science',
                       'graduation_date': '2026-12', 'graduation_status': 'expected'},
        'availability': {'can_relocate': True, 'relocation_locations': ['New York City'],
                         'start_date_preference': 'As early as possible after graduation'},
        'application_workflow': {'unknown_required_answer': 'skip_application'},
    }


def test_context_keeps_zero_experience_and_expected_degree(applicant):
    text = prompt._build_profile_summary(applicant)
    assert 'Years Experience: 0' in text
    assert 'Degree status: expected' in text
    assert 'Graduation: 2026-12' in text
    assert 'Future Sponsorship Needed: False' in text
    assert 'Citizenship: United States' in text


def test_relocation_does_not_expand_user_preference(applicant):
    text = prompt._build_screening_section(applicant)
    assert 'only to New York City' in text
    assert 'for the right role' not in text


@pytest.mark.parametrize('dry_run', [False, True])
def test_assembled_prompt_supports_writing_and_bounded_repair(applicant, tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(prompt.config, 'load_profile', lambda: applicant)
    monkeypatch.setattr(prompt.config, 'load_search_config', lambda: {})
    monkeypatch.setattr(prompt.config, 'load_blocked_sso', lambda: [])
    monkeypatch.setattr(prompt.config, 'APPLY_WORKER_DIR', tmp_path / 'workers')
    resume = tmp_path / 'resume.txt'
    resume.with_suffix('.pdf').write_bytes(b'%PDF-test-fixture')
    text = prompt.build_prompt({'title': 'Junior Engineer', 'company': 'Example',
                               'url': 'https://example.com/jobs/1',
                               'tailored_resume_path': str(resume)},
                              'Documented engineering experience', dry_run=dry_run)
    assert 'GENERATED RESPONSE' in text
    assert 'required narrative responses' in text
    assert 'VALIDATION RECOVERY' in text
    assert 'actual employment entry' in text
    assert 'Never fill an arbitrary day' in text
    assert 'ALWAYS upload fresh' not in text
    assert 'TAILORED RESUME summary' not in text
    assert 'Do whatever it takes' not in text
    assert 'Do not manufacture a calendar day' in text
    assert 'exact field/error text' in text
    assert 'verify trusted human authorization' in text
    assert ('dry_run_not_submitted' in text) is dry_run
    assert (tmp_path / 'workers/current/Test_Candidate_Resume.pdf').read_bytes() == b'%PDF-test-fixture'


def test_posted_salary_policy_does_not_invent_a_floor(applicant):
    applicant['compensation'] = {
        'salary_expectation': 'Flexible within the posted range',
        'posted_range_policy': 'Use the posted midpoint if numeric; hold if no range.'}
    text = prompt._build_salary_section(applicant)
    assert 'hold if no range' in text
    assert 'FLOOR' not in text
