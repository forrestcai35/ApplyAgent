"""Read-only Codex/Playwright integration smoke test; submits no application."""
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from applyagent.apply.backend import run_codex
from applyagent.apply.chrome import _kill_process_tree
from applyagent.config import LOG_DIR, get_chrome_path, load_env

load_env()
LOG_DIR.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='applyagent-browser-check-') as folder:
    root = Path(folder)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    browser = subprocess.Popen([get_chrome_path(), '--headless=new', '--no-first-run',
        f'--user-data-dir={root / "browser"}', f'--remote-debugging-port={port}', 'about:blank'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        time.sleep(3)
        result = run_codex(
            'Read-only integration test. Use browser_navigate to open https://example.com and '
            'browser_snapshot to read it. Do not fill forms, upload data, or submit anything. '
            'If the page title is Example Domain and you read the page body, output RESULT:FAILED:browser_smoke_ok. '
            'Otherwise return RESULT:FAILED:browser_smoke_failed. Include the observed title and a short body excerpt in your explanation.',
            port, root, LOG_DIR / 'codex-browser-smoke.jsonl', None, True, 120,
            lambda proc: None, lambda: None)
        print(result)
        output = root / 'codex-result.txt'
        if output.exists():
            print(output.read_text())
        raise SystemExit(0 if result == 'failed:browser_smoke_ok' else 1)
    finally:
        _kill_process_tree(browser.pid)
        browser.wait()
