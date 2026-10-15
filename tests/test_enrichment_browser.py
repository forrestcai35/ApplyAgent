from unittest.mock import Mock

import pytest

from applyagent.enrichment import browser


def test_native_browser_keeps_proxy_settings_and_closes_on_failure():
    playwright = Mock()
    options = {'headless': True, 'executable_path': '/apps/Brave Browser',
               'proxy': {'server': 'http://proxy.example'}}
    with pytest.raises(ValueError):
        with browser.posting_browser(playwright, options):
            raise ValueError('extraction failed')
    playwright.chromium.launch.assert_called_once_with(**options, timeout=30_000)
    playwright.chromium.launch.return_value.close.assert_called_once()
    playwright.chromium.connect_over_cdp.assert_not_called()


def prepare_cdp(monkeypatch):
    sock = Mock()
    sock.getsockname.return_value = ('127.0.0.1', 9555)
    socket_manager = Mock()
    socket_manager.__enter__ = Mock(return_value=sock)
    socket_manager.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(browser.socket, 'socket', lambda: socket_manager)
    process = Mock(pid=321)
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(browser.subprocess, 'Popen', spawn)
    client = Mock()
    response = client.get.return_value
    response.status_code = 200
    response.json.return_value = {'webSocketDebuggerUrl': 'ws://127.0.0.1:9555/devtools/browser/test'}
    manager = Mock()
    manager.__enter__ = Mock(return_value=client)
    manager.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(browser.httpx, 'Client', lambda **kw: manager)
    kill = Mock()
    monkeypatch.setattr(browser, '_kill_process_tree', kill)
    return spawn, process, client, kill


def test_brave_uses_fresh_profile_and_owned_cdp_process(monkeypatch):
    spawn, process, client, kill = prepare_cdp(monkeypatch)
    playwright = Mock()
    with browser.posting_browser(playwright, {'headless': True, 'executable_path': '/apps/Brave Browser'}) as connected:
        assert connected is playwright.chromium.connect_over_cdp.return_value
    playwright.chromium.launch.assert_not_called()
    playwright.chromium.connect_over_cdp.assert_called_once_with('http://127.0.0.1:9555', timeout=10_000)
    args = spawn.call_args.args[0]
    assert '--remote-debugging-address=127.0.0.1' in args
    assert any(arg.startswith('--user-data-dir=') for arg in args)
    assert '--no-sandbox' not in args
    client.get.assert_called_once_with('http://127.0.0.1:9555/json/version')
    kill.assert_called_once_with(321)
    process.wait.assert_called_once_with(timeout=5)


def test_cdp_connection_failure_cleans_up_test_owned_browser(monkeypatch):
    _, process, _, kill = prepare_cdp(monkeypatch)
    playwright = Mock()
    playwright.chromium.connect_over_cdp.side_effect = RuntimeError('CDP unreachable')
    with pytest.raises(RuntimeError, match='unreachable'):
        with browser.posting_browser(playwright, {'headless': True, 'executable_path': '/apps/Brave Browser'}):
            pytest.fail('connection should not yield')
    kill.assert_called_once_with(321)
    process.wait.assert_called_once_with(timeout=5)
