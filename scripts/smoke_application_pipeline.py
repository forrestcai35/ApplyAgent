"""Live local pipeline check with synthetic data, never real employer submissions.

Run with the project's Python and --output-dir pointing at an isolated directory.
Artifacts and synthetic receipts stay there, outside the production queue.
"""

import argparse
import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def fixture_pdf(path):
    content = b'BT /F1 14 Tf 50 750 Td (Synthetic Pipeline Test Candidate - Python React SQL) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(content)).encode() + b' >>\nstream\n' + content + b'\nendstream']
    data = b'%PDF-1.4\n'
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f'{number} 0 obj\n'.encode() + obj + b'\nendobj\n'
    xref = len(data)
    data += b'xref\n0 6\n0000000000 65535 f \n'
    data += b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets[1:])
    data += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['codex'], default='codex')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh output directory; do not overwrite or resume prior fixtures')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['APPLYAGENT_DIR'] = str(root)
    os.environ['APPLYAGENT_LLM_BACKEND'] = args.backend
    os.environ['APPLYAGENT_APPLY_BACKEND'] = args.backend
    os.environ['APPLYAGENT_APPLY_TIMEOUT'] = '240'
    os.environ['APPLYAGENT_HARNESS_TIMEOUT'] = '120'

    from applyagent import config
    from applyagent.apply import launcher
    from applyagent.apply.refill import replenish
    from applyagent.apply.runtime import daily_budget
    from applyagent.database import init_db
    from applyagent.scheduled import run_scheduled

    config.ensure_dirs()
    # A pre-created empty profile prevents copying the user's browser cookies.
    (config.CHROME_WORKER_DIR / 'worker-0' / 'Default').mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        launcher.BASE_CDP_PORT = sock.getsockname()[1]
    posts = []
    descriptions = 'Full-time Junior Software Engineer in New York, NY. Build Python APIs, React interfaces and PostgreSQL data services. New graduates with a Computer Science degree in progress are welcome. Internship and project experience are sufficient; no fixed years minimum. Work with mentors to build and test reliable software. Base salary $100,000-$130,000. Start after expected graduation.'
    server = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def page(self, body, code=200):
            payload = ('<!doctype html><html><head><title>Local Pipeline Fixture</title></head><body>'
                       + body + '</body></html>').encode()
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/jobs':
                nodes = [posting(kind) for kind in ['unknown', 'complete']]
                return self.page('<h1>Synthetic job board</h1><script type="application/ld+json">'
                                 + json.dumps(nodes) + '</script>')
            if self.path.startswith('/job/'):
                kind = self.path.rsplit('/', 1)[-1]
                return self.page('<h1>Junior Software Engineer — Fixture Engineering</h1><p>' + descriptions
                                 + '</p><a href="/apply/' + kind + '">Apply for this Job</a>'
                                 + '<script type="application/ld+json">' + json.dumps(posting(kind)) + '</script>')
            if self.path.startswith('/apply/'):
                kind = self.path.rsplit('/', 1)[-1]
                extra = '<label>Cumulative GPA (required)<input name="gpa" type="number" step="0.01" required></label>' if kind == 'unknown' else '<label>Cumulative GPA (optional)<input name="gpa" type="number" step="0.01"></label>'
                return self.page('<h1>Junior Software Engineer — Fixture Engineering</h1><p>' + descriptions
                    + '</p><p>Local synthetic pipeline fixture; no employment application is being sent.</p>'
                    + f'<form method="post" enctype="multipart/form-data" action="/submit/{kind}">'
                    + '<label>Full name<input name="name" required></label>'
                    + '<label>Email<input name="email" type="email" required></label>'
                    + '<label>Phone<input name="phone" type="tel" required></label>'
                    + '<label>Resume<input type="file" name="resume" accept=".pdf" required></label>'
                    + '<label>Why Fixture Engineering?<textarea name="motivation" required></textarea></label>'
                    + '<label>Authorized to work in the United States?<select name="authorized" required><option value="">Select</option><option value="yes">Yes</option><option value="no">No</option></select></label>'
                    + '<label>Require sponsorship now or in the future?<select name="sponsorship" required><option value="">Select</option><option value="yes">Yes</option><option value="no">No</option></select></label>'
                    + extra + '<button type="submit">Submit Application</button></form>')
            if self.path == '/confirmation':
                return self.page('<h1>Your application was successfully submitted.</h1><p>Fixture Engineering synthetic test only.</p>')
            self.page('<h1>Not found</h1>', 404)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            message = BytesParser(policy=policy.default).parsebytes(
                ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + raw)
            fields, upload = {}, None
            for part in message.iter_parts():
                name = part.get_param('name', header='content-disposition')
                payload = part.get_payload(decode=True)
                if name == 'resume':
                    upload = {'filename': part.get_filename(), 'sha256': hashlib.sha256(payload).hexdigest()}
                else:
                    fields[name] = payload.decode()
            posts.append({'path': self.path, 'fields': fields, 'upload': upload})
            required = ['name', 'email', 'phone', 'motivation', 'authorized', 'sponsorship']
            if not all(fields.get(key) for key in required) or not upload:
                return self.page('<h1>Required fields missing; application was not accepted.</h1>', 400)
            self.send_response(303)
            self.send_header('Location', '/confirmation')
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    base = f'http://127.0.0.1:{server.server_port}'

    def posting(kind):
        return {'@context': 'https://schema.org', '@type': 'JobPosting',
                'title': 'Junior Software Engineer', 'description': descriptions,
                'hiringOrganization': {'name': 'Fixture Engineering'},
                'jobLocation': {'address': {'addressLocality': 'New York', 'addressRegion': 'NY'}},
                'url': base + '/job/' + kind,
                'applicationContact': {'url': base + '/apply/' + kind}}

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = {'personal': {'full_name': 'Pipeline Test Candidate', 'email': 'candidate@example.com',
               'phone': '2025550123', 'city': 'San Jose', 'province_state': 'CA', 'country': 'United States'},
               'work_authorization': {'legally_authorized_to_work': True, 'require_sponsorship': False,
                                      'future_sponsorship_required': False, 'citizenship': 'United States'},
               'experience': {'education_level': 'BS Computer Science expected December 2026',
                              'school': 'Fixture University', 'graduation_date': '2026-12', 'graduation_status': 'expected'},
               'availability': {'can_relocate': True, 'relocation_locations': ['New York City'],
                                'start_date_preference': 'As early as possible consistent with graduation'},
               'application_workflow': {'unknown_required_answer': 'skip_application',
                                        'daily_submission_cap': 1, 'timezone': 'America/Los_Angeles'}}
    (root / 'profile.json').write_text(json.dumps(profile))
    (root / 'resume.txt').write_text('SYNTHETIC TEST DATA. Pipeline Test Candidate. Fixture University BS Computer Science expected December 2026. Fixture Software Intern May 2025-August 2025: built Python REST APIs and optimized PostgreSQL queries. Project: built a React interface and Python service with SQL storage and automated tests. Skills: Python, TypeScript, React, PostgreSQL, Git. No GPA supplied.')
    fixture_pdf(root / 'resume.pdf')
    (root / 'searches.yaml').write_text(json.dumps({'discovery_sources': ['public'],
        'public_pages': [{'name': 'Local Fixture', 'url': base + '/jobs'}],
        'quality': {'enabled': True, 'min_score': 60, 'shortlist_limit': 10}}))
    os.environ['APPLYAGENT_TASK_INSTRUCTION'] = (
        'The user requested a pipeline test. This invocation uses only synthetic test data and an isolated local fixture. '
        f'Operate only on {base}/job/*, {base}/apply/*, {base}/submit/* and {base}/confirmation for Fixture Engineering. '
        'Fill and submit the synthetic local test form when required answers are supported. Stop at required unknowns. '
        'There is no real applicant data or real employer application. Do not navigate to any external site. '
        'Treat a local fixture confirmation as the test result; only the isolated test registry will record it.')
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'synthetic_only': True,
              'real_employer_submissions': 0, 'stages': {}, 'artifacts': str(root)}
    conn = init_db()
    try:
        print('Discovering synthetic local postings', flush=True)
        code = run_scheduled(timeout=60)
        report['stages']['discovery'] = {'exit_code': code, 'jobs': conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]}
        if code or report['stages']['discovery']['jobs'] != 2:
            raise RuntimeError('Local discovery did not collect both fixtures')
        print('Running real enrichment and configured model scoring', flush=True)
        prepared = replenish(batch_size=2, max_candidates=2, discover=False)
        report['stages']['enrichment_and_scoring'] = {'prepared': prepared,
            'roles': [dict(row) for row in conn.execute('SELECT url,fit_score,detail_error FROM jobs')]}
        if prepared != 2:
            raise RuntimeError('Both synthetic roles must pass enrichment and scoring')
        print('Checking mandatory-unknown preflight with real browser agent', flush=True)
        launcher.main(limit=1, target_url=base + '/job/unknown', headless=True, backend=args.backend, refill=False)
        unknown = dict(conn.execute("SELECT apply_status,apply_error,applied_at FROM jobs WHERE url=?", (base + '/job/unknown',)).fetchone())
        report['stages']['required_unknown'] = unknown
        if unknown['apply_error'] != 'unknown_required_answer' or posts:
            raise RuntimeError('Required-unknown fixture did not stop before submission')
        print('Checking local synthetic submission and receipt persistence', flush=True)
        launcher.main(until_submitted=1, headless=True, backend=args.backend, refill=False)
        budget = daily_budget()
        report['stages']['synthetic_submission'] = {'submitted': budget.submitted, 'remaining': budget.remaining}
        if budget.submitted != 1 or len(posts) != 1 or posts[0]['path'] != '/submit/complete':
            raise RuntimeError('Expected exactly one confirmed local synthetic submission')
        sent = posts[0]
        if sent['fields'].get('gpa') or sent['fields'].get('authorized') != 'yes' or sent['fields'].get('sponsorship') != 'no':
            raise RuntimeError('Optional unknown or work-authorization answer was incorrect')
        if sent['upload']['sha256'] != hashlib.sha256((root / 'resume.pdf').read_bytes()).hexdigest():
            raise RuntimeError('Uploaded resume differs from synthetic fixture')
        if not (root / 'already_applied.json').exists() or not list((root / 'reports').glob('*-submitted-*.json')):
            raise RuntimeError('Submission registry or receipt missing')
        print('Checking shared daily cap on a second run', flush=True)
        launcher.main(until_submitted=1, headless=True, backend=args.backend, refill=False)
        if daily_budget().submitted != 1 or len(posts) != 1:
            raise RuntimeError('Second run exceeded the synthetic daily submission cap')
        report['stages']['daily_cap_restart'] = {'submitted': 1, 'extra_form_posts': 0}
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (root / 'form-posts.json').write_text(json.dumps(posts, indent=2))
        (root / 'smoke-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
