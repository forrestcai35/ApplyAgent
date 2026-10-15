"""Local review holds and a compact queue. Saved reports never grant permission."""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta

from applyagent import config
from applyagent.applications import application_identity


def _profile_sections(status: str) -> tuple[str, ...]:
    if 'experience' in status:
        return ('experience', 'resume_facts', 'skills_boundary')
    return ('personal', 'experience', 'resume_facts', 'availability',
            'work_authorization', 'skills_boundary', 'compensation',
            'voluntary_self_identification')


def _profile_hash(sections: tuple[str, ...] | None = None) -> str:
    profile = config.load_profile()
    # Timeouts, attempt caps and prompt wording are operational settings, not
    # new applicant facts that resolve a hold.
    facts = {key: profile.get(key) for key in (sections or _profile_sections(''))}
    if isinstance(facts.get('availability'), dict):
        facts['availability'] = {key: value for key, value in facts['availability'].items()
                                 if not key.startswith('interview_process_timing')}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def _load() -> dict:
    path = config.APP_DIR / 'review-state.json'
    state = json.loads(path.read_text()) if path.exists() else {'entries': {}}
    shared = {}
    for key, record in state['entries'].items():
        if record.get('profile_hash_version') != 3:
            # Conservatively preserve existing evidence during scope migration.
            record['profile_hash'] = _profile_hash(_profile_sections(record['status']))
            record['profile_hash_version'] = 3
        # JSON does not preserve shared references for exact URL aliases.
        signature = json.dumps(record, sort_keys=True)
        state['entries'][key] = shared.setdefault(signature, record)
    return state


def _save(state: dict) -> None:
    path = config.APP_DIR / 'review-state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def remember(entry: dict, *, state: dict | None = None) -> dict | None:
    """Store negative review evidence only, keyed by observed URL aliases."""
    status = entry.get('status', '')
    if not any(word in status for word in ('held', 'excluded', 'closed', 'unavailable')):
        return None
    urls = [entry.get(key) for key in ('employer_application_url', 'application_url', 'url', 'source_url')]
    keys = list(dict.fromkeys(application_identity(url) for url in urls if url))
    if not keys:
        return None
    current = state if state is not None else _load()
    # Updating a package must update its older board aliases too, rather than
    # leaving a second stale hold for the same observed employer requisition.
    for key in keys:
        previous = current['entries'].get(key)
        if previous:
            urls.extend(url for url in previous['urls'] if url not in urls)
    urls = list(dict.fromkeys(url for url in urls if url))
    keys = [application_identity(url) for url in urls]
    record = {
        'company': entry.get('company'), 'title': entry.get('title') or entry.get('role'),
        'status': status, 'reason': entry.get('reason') or '; '.join(entry.get('unresolved', [])),
        'mandatory_unknowns': entry.get('mandatory_unknowns', []),
        'checked_at': entry.get('checked_at') or datetime.now(UTC).isoformat(),
        'profile_hash': _profile_hash(_profile_sections(status)), 'urls': urls,
        'profile_hash_version': 3,
        'authorization': 'Not established by this cache; trusted human evidence is still required',
    }
    for key in keys:
        current['entries'][key] = record
    if state is None:
        _save(current)
    return record


def refresh_from_reports() -> dict:
    """Import recent negative reviews, then richer packages, once per preparation run."""
    state = _load()
    reports = config.APP_DIR / 'reports'
    # Dated workflow reports only: never import synthetic fixtures or agent logs.
    paths = sorted(reports.glob('applications-????-??-??.json'))[-7:]
    paths += sorted(reports.glob('application-packages-????-??-??.json'))[-7:]
    for path in paths:
        report = json.loads(path.read_text())
        sources = {job['url']: job for job in report.get('discovery', {}).get('new_jobs', []) if job.get('url')}
        for key in ('review', 'supplemental_review', 'packages'):
            for entry in report.get(key, []):
                entry = dict(entry)
                source = sources.get(entry.get('source_url') or entry.get('url'))
                if source and source.get('application_url'):
                    entry.setdefault('application_url', source['application_url'])
                urls = [entry.get(k) for k in ('employer_application_url', 'application_url', 'url', 'source_url')]
                existing = next((state['entries'].get(application_identity(u)) for u in urls
                                 if u and application_identity(u) in state['entries']), None)
                # Do not refresh unchanged timestamps/profile hashes every day:
                # changing the profile should make fact holds reviewable again.
                checked = entry.get('checked_at') or report.get('updated_at', '')
                # A ledger's updated_at also changes when a receipt/run is added.
                # An undated summary must not replace richer requisition evidence.
                if existing and (not entry.get('checked_at') or existing['checked_at'] >= checked):
                    if existing['checked_at'] == checked and entry.get('mandatory_unknowns'):
                        existing['mandatory_unknowns'] = entry['mandatory_unknowns']
                    for url in urls:
                        if url and url not in existing['urls']:
                            existing['urls'].append(url)
                            state['entries'][application_identity(url)] = existing
                    continue
                remember(dict(entry, checked_at=checked), state=state)
    _save(state)
    return state


def deferred_reason(job: dict, *, state: dict | None = None, now: datetime | None = None) -> str | None:
    state = state if state is not None else _load()
    record = next((state['entries'].get(application_identity(job[key])) for key in
                   ('application_url', 'employer_application_url', 'url', 'source_url')
                   if job.get(key) and application_identity(job[key]) in state['entries']), None)
    if not record:
        return None
    excluded = excluded_required_fields(record['mandatory_unknowns'])
    if excluded:
        return 'Applicant excludes mandatory fields: ' + '; '.join(excluded)
    # Closed/mismatch/unverified pages may change; retry local verification weekly.
    # Never age out approval, legal, uncertain submission or duplicate holds.
    reason = record['reason'].lower()
    protected = any(term in reason for term in ('authorization', 'approval', 'terms', 'duplicate', 'uncertain submission'))
    if not protected:
        checked = datetime.fromisoformat(record['checked_at'])
        if (now or datetime.now(UTC)) - checked >= timedelta(days=7):
            return None
        if record['profile_hash'] != _profile_hash(_profile_sections(record['status'])):
            return None
    return record['reason'] or record['status']


def excluded_required_fields(labels: list[str], profile: dict | None = None) -> list[str]:
    """Exact required-field labels, never optional prose or loose keyword guesses."""
    profile = profile if profile is not None else config.load_profile()
    excluded = profile.get('application_workflow', {}).get('skip_required_fields', [])
    patterns = {'gpa': r'\bgpa\b|grade point average', 'transcript': r'\btranscripts?\b'}
    return [label for label in labels if any(re.search(patterns[key], label, re.IGNORECASE)
            for key in excluded if key in patterns)]


def queue_summary(refresh: bool = False) -> dict:
    state = refresh_from_reports() if refresh else _load()
    unique = {json.dumps(record, sort_keys=True): record for record in state['entries'].values()}
    items = []
    for record in unique.values():
        labels = record['mandatory_unknowns']
        excluded = excluded_required_fields(labels)
        items.append({**record, 'queue_status': 'excluded_by_preference' if excluded else record['status'],
                      'excluded_fields': excluded})
    return {'items': items, 'held_requisitions': len(items),
            'excluded_by_preference': sum(bool(item['excluded_fields']) for item in items),
            'submission_authorization': 'No authorization is granted or verified by this summary'}
