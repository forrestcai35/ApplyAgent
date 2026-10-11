import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from applyagent.harness import HarnessClient, selected_backend


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('APPLYAGENT_LLM_BACKEND', 'claude')
    monkeypatch.setattr('applyagent.harness.shutil.which', lambda _: '/bin/claude')
    return HarnessClient()


def test_harness_is_default_without_api(monkeypatch):
    for key in ('APPLYAGENT_LLM_BACKEND', 'LLM_URL', 'GEMINI_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    assert selected_backend() == 'claude'
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    assert selected_backend() == 'api'
    monkeypatch.setenv('APPLYAGENT_LLM_BACKEND', 'claude')
    assert selected_backend() == 'claude'


def test_text_result_and_scoped_subprocess(client):
    proc = Mock(returncode=0)
    proc.communicate.return_value = (json.dumps({'result': 'SCORE: 8', 'is_error': False}), '')
    with patch('applyagent.harness.subprocess.Popen', return_value=proc) as spawn:
        assert client.ask('private resume') == 'SCORE: 8'
    args = spawn.call_args.args[0]
    assert args[args.index('--tools') + 1] == ''
    assert '--strict-mcp-config' in args
    assert '--no-session-persistence' in args
    assert 'private resume' not in str(args)
    assert 'private resume' in proc.communicate.call_args.args[0]


@pytest.mark.parametrize('response', ['bad json', '{}', '{"is_error":true,"result":"error"}', '{"result":"", "subtype":"success"}', '[]'])
def test_rejects_bad_results(client, response):
    proc = Mock(returncode=0)
    proc.communicate.return_value = (response, '')
    with patch('applyagent.harness.subprocess.Popen', return_value=proc), pytest.raises(RuntimeError):
        client.ask('test')


def test_nonzero_exit(client):
    proc = Mock(returncode=1)
    proc.communicate.return_value = ('', 'private provider detail')
    with patch('applyagent.harness.subprocess.Popen', return_value=proc), pytest.raises(RuntimeError, match='code 1'):
        client.ask('test')


def test_timeout_reaps_process_group(client):
    proc = Mock(pid=123)
    proc.communicate.side_effect = [subprocess.TimeoutExpired('claude', 1), ('', '')]
    with patch('applyagent.harness.subprocess.Popen', return_value=proc), patch('applyagent.harness.os.killpg') as kill:
        with pytest.raises(RuntimeError, match='timed out'):
            client.ask('test')
        kill.assert_called_once()
        assert proc.communicate.call_count == 2


def test_get_client_routes_to_harness(monkeypatch):
    from applyagent import llm
    monkeypatch.setenv('APPLYAGENT_LLM_BACKEND', 'claude')
    monkeypatch.setattr('applyagent.harness.shutil.which', lambda _: '/bin/claude')
    monkeypatch.setattr(llm, '_instance', None)
    assert isinstance(llm.get_client(), HarnessClient)


def test_codex_reads_final_message(monkeypatch, client):
    from pathlib import Path
    client.engine = 'codex'
    proc = Mock(returncode=0)
    def start(cmd, **kwargs):
        Path(cmd[cmd.index('--output-last-message') + 1]).write_text('SCORE: 9')
        assert '--ignore-user-config' in cmd
        assert cmd[cmd.index('--sandbox') + 1] == 'read-only'
        assert 'features.shell_tool=false' in cmd
        proc.communicate.return_value = ('{"type":"turn.completed"}', '')
        return proc
    with patch('applyagent.harness.subprocess.Popen', side_effect=start):
        assert client.ask('score this') == 'SCORE: 9'
