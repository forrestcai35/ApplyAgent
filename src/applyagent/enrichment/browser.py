"""Bounded, isolated browser sessions for public posting enrichment."""

import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from applyagent.apply.chrome import _kill_process_tree


@contextmanager
def posting_browser(playwright, launch_options):
    """Brave works over CDP even when Playwright's debug pipe fails to start.

    Keep explicit proxy configuration on Playwright's native launch path. CDP
    uses a fresh profile, a loopback endpoint and only the owned process tree.
    """
    executable = launch_options.get('executable_path', '')
    if Path(executable).name != 'Brave Browser' or launch_options.get('proxy'):
        browser = playwright.chromium.launch(**launch_options, timeout=30_000)
        try:
            yield browser
        finally:
            browser.close()
        return

    with tempfile.TemporaryDirectory(prefix='applyagent-enrichment-browser-') as folder:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        endpoint = f'http://127.0.0.1:{port}'
        args = [executable, f'--user-data-dir={folder}', f'--remote-debugging-port={port}',
                '--remote-debugging-address=127.0.0.1', '--no-first-run',
                '--no-default-browser-check', 'about:blank']
        if launch_options.get('headless'):
            args.insert(1, '--headless=new')
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        browser = None
        try:
            deadline = time.monotonic() + 20
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Enrichment browser exited before its CDP endpoint opened')
                    try:
                        response = client.get(endpoint + '/json/version')
                        if response.status_code == 200 and response.json().get('webSocketDebuggerUrl'):
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError('Enrichment browser did not open its CDP endpoint within 20 seconds')
            browser = playwright.chromium.connect_over_cdp(endpoint, timeout=10_000)
            yield browser
        finally:
            try:
                if browser is not None:
                    browser.close()
            finally:
                if process.poll() is None:
                    _kill_process_tree(process.pid)
                process.wait(timeout=5)
