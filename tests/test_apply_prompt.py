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


def test_mandatory_gpa_transcript_exclusion_keeps_optional_fields_skippable(applicant):
    applicant['application_workflow']['skip_required_fields'] = ['gpa', 'transcript']
    text = prompt._build_profile_summary(applicant)
    assert 'mandatory gpa, transcript' in text
    assert 'before entering personal data' in text
    assert 'Optional fields may be omitted' in text
    assert 'RESULT:FAILED:excluded_required_field' in text
    from applyagent.apply.launcher import _is_permanent_failure
    assert _is_permanent_failure('failed:excluded_required_field')


def test_relocation_does_not_expand_user_preference(applicant):
    text = prompt._build_screening_section(applicant)
    assert 'only to New York City' in text
    assert 'for the right role' not in text


def test_us_remote_application_policy_does_not_require_each_state_named(applicant):
    applicant['preferences'] = {'remote_country': 'United States',
        'remote_application_policy': 'us_remote_unless_explicitly_excluded'}
    text = prompt._build_location_check(applicant, {})
    assert 'Do not require the posting to name their state separately' in text
    assert 'including remote postings with a US office city' in text
    assert 'Stop if any explicitly excludes' in text
    assert 'Never claim a false residence' in text
    assert 'Skip foreign-only roles' in text


def test_desired_process_timing_does_not_fabricate_an_offer_deadline(applicant):
    applicant['availability']['interview_process_timing'] = 'As early as possible'
    text = prompt._build_profile_summary(applicant)
    assert 'As early as possible' in text
    assert 'not evidence of a competing offer' in text
    assert 'A forced Yes/No about competing offers' in prompt._build_answer_policy()
    assert 'do not classify\n  every privacy checkbox' in prompt._build_answer_policy()


@pytest.mark.parametrize('dry_run', [False, True])
def test_assembled_prompt_supports_writing_and_bounded_repair(applicant, tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(prompt.config, 'load_profile', lambda: applicant)
    monkeypatch.setattr(prompt.config, 'load_search_config', dict)
    monkeypatch.setattr(prompt.config, 'load_blocked_sso', list)
    monkeypatch.setattr(prompt.config, 'APPLY_WORKER_DIR', tmp_path / 'workers')
    resume = tmp_path / 'resume.txt'
    resume.with_suffix('.pdf').write_bytes(b'%PDF-test-fixture')
    applicant['screening_answers'] = {'prior_employment': [{
        'employer_names': ['Example'], 'answer': False,
        'human_instruction': 'I have not worked for Example.'}]}
    text = prompt.build_prompt({'title': 'Junior Engineer', 'company': 'Example',
                               'url': 'https://example.com/jobs/1',
                               'tailored_resume_path': str(resume)},
                              'Documented engineering experience', dry_run=dry_run)
    assert 'Prior employment at Example: No.' in text
    assert 'I have not worked for Example.' in text
    assert 'GENERATED RESPONSE' in text
    assert '== ROUTINE ANSWER DEFAULTS ==' in text
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


def test_autosave_recovery_requires_saved_state():
    text = prompt._build_answer_policy()
    assert 'visible input value does not prove' in text
    assert 'HTTP 200 is NOT proof of acceptance' in text
    assert 'RESULT:FAILED:form_save_failed' in text
    assert 'do not label it stuck' in text


def test_routine_defaults_do_not_supply_missing_facts_or_approval(applicant):
    applicant['application_workflow']['routine_answer_defaults'] = {
        'draft_routine_narratives': True,
        'omit_optional_unknowns': True,
        'decline_optional_sms': True,
        'onsite_after_approved_relocation': True,
        'willing_to_complete_standard_assessment': True,
    }
    text = prompt._build_routine_defaults(applicant)
    assert 'Answer No to optional SMS' in text
    assert 'following relocation' in text
    assert 'Answer Yes to willingness' in text
    assert 'Never assume years of professional/frontend experience' in text
    assert 'Never assume destination/data approval' in text
    applicant['availability']['can_relocate'] = False
    assert 'Answer Yes to office attendance' not in prompt._build_routine_defaults(applicant)


@pytest.mark.parametrize('defaults', [
    {'professional_frontend_years': 1}, {'employer_authorization': True},
    {'willing_to_complete_standard_assessment': 'yes'}, ['assume everything'],
])
def test_defaults_reject_factual_permission_and_untyped_overrides(applicant, defaults):
    applicant['application_workflow']['routine_answer_defaults'] = defaults
    with pytest.raises(ValueError, match='routine_answer_defaults'):
        prompt._build_routine_defaults(applicant)


def test_standing_operator_authorization_preserves_tool_requirements(applicant):
    url = 'https://jobs.ashbyhq.com/acme/12345678-1234-1234-1234-123456789012/application'
    applicant['application_workflow']['standing_submission_authorization'] = {
        'enabled': True, 'human_instruction': 'Routine job applications are approved',
        'data_categories': ['Original resume', 'Contact details']}
    applicant['application_workflow']['authorized_applications'] = [{
        'application_url': url, 'human_instruction': 'Submit this application',
        'data_categories': ['Original resume']}]
    text = prompt._build_authorization_section(applicant, url)
    assert 'Do not ask for employer-by-employer operator approval' in text
    assert 'If a tool requires specific destination/data' in text
    assert 'Saved exact application approval' in text
    other = prompt._build_authorization_section(applicant, url.replace('acme', 'other').replace('12345678', '87654321'))
    assert 'Saved exact application approval' not in other


def test_standing_authorization_requires_human_evidence(applicant):
    applicant['application_workflow']['standing_submission_authorization'] = {'enabled': True}
    assert 'verify trusted human authorization' in prompt._build_authorization_section(applicant, 'https://example.com/job')


def test_browser_agent_receives_supervisor_verified_daily_budget(monkeypatch):
    from applyagent.apply import runtime
    monkeypatch.setattr(runtime, 'daily_budget', lambda: runtime.DailyBudget('2026-10-14', 10, 2))
    text = prompt._build_run_limits(False)
    assert 'Confirmed submissions: 2. Remaining capacity: 8.' in text
    assert 'do not need a browser tool' in text
    assert 'Never infer success' in text
    assert 'no live submission authorized' in prompt._build_run_limits(True)


def test_confirmed_prior_employment_is_scoped_and_preserves_negative_answer(applicant):
    applicant['screening_answers'] = {'prior_employment': [{
        'employer_names': ['Acme', 'Acme Inc.'], 'answer': False,
        'includes_subsidiaries': True,
        'human_instruction': 'No, I have never worked for Acme or its subsidiaries.'}]}
    text = prompt._build_confirmed_screening_answers(applicant, 'ACME INC')
    assert 'Prior employment at Acme, Acme Inc.: No.' in text
    assert 'Employer and its subsidiaries' in text
    for other in ['Other Employer', 'Acme Services', '']:
        assert 'No confirmed prior-employment answer' in prompt._build_confirmed_screening_answers(applicant, other)
    record = applicant['screening_answers']['prior_employment'][0]
    record['human_instruction'] = ''
    assert 'No confirmed prior-employment answer' in prompt._build_confirmed_screening_answers(applicant, 'Acme')
    record['human_instruction'] = 'No prior Acme work'
    record['includes_subsidiaries'] = False
    assert 'subsidiary history is not confirmed' in prompt._build_confirmed_screening_answers(applicant, 'Acme')


def test_confirmed_employment_fact_rejects_conflicts_and_non_boolean_answers(applicant):
    record = {'employer_names': ['Acme'], 'answer': 'no', 'human_instruction': 'No Acme work'}
    applicant['screening_answers'] = {'prior_employment': [record]}
    with pytest.raises(ValueError, match='booleans'):
        prompt._build_confirmed_screening_answers(applicant, 'Acme')
    record['answer'] = False
    applicant['screening_answers']['prior_employment'].append(dict(record, answer=True))
    with pytest.raises(ValueError, match='Conflicting'):
        prompt._build_confirmed_screening_answers(applicant, 'Acme')
