"""Selection and bounded execution of the Codex browser application backend."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def selected_backend(override: str | None = None) -> str:
    backend = override or os.environ.get("APPLYAGENT_APPLY_BACKEND", "auto")
    if backend == "auto":
        from applyagent.harness import selected_backend as text_backend
        backend = text_backend()
        if backend not in {"codex", "claude"}:
            backend = "codex" if shutil.which("codex") else "claude"
    if backend not in {"codex", "claude"}:
        raise ValueError("Application backend must be auto, codex, or claude")
    return backend


def executable(backend: str) -> str | None:
    return shutil.which(os.environ.get(f"APPLYAGENT_{backend.upper()}_BIN", backend))


def codex_command(port: int, output: Path, model: str | None = None) -> list[str]:
    binary = executable("codex")
    if not binary:
        raise RuntimeError("Codex CLI missing; install Codex and sign in")
    args = [binary, "--no-daemon", "exec", "--ignore-user-config", "--strict-config",
            "--skip-git-repo-check", "--ephemeral", "--approve-for-me",
            "-c", 'web_search="disabled"',
            "-c", "features.shell_tool=false", "-c", "features.apps=false",
            "-c", "features.multi_agent=false",
            "-c", 'mcp_servers.playwright.command="npx"',
            "-c", "mcp_servers.playwright.required=true",
            "-c", "mcp_servers.playwright.startup_timeout_sec=60",
            "-c", "mcp_servers.playwright.args=" + json.dumps([
                "-y", "@playwright/mcp@latest", f"--cdp-endpoint=http://localhost:{port}"]),
            "--json", "--output-last-message", str(output)]
    if model:
        args += ["--model", model]
    return args + ["-"]


def parse_result(output: str, dry_run: bool = False) -> str:
    # Only the final message is parsed, never echoed prompts or tool output.
    results = re.findall(r"^RESULT:(APPLIED|EXPIRED|CAPTCHA|LOGIN_ISSUE|FAILED:[^\n]+)\s*$",
                         output, re.MULTILINE)
    if len(results) != 1:
        return "failed:submission_uncertain"
    result = results[0].strip().lower()
    if result == "applied":
        if dry_run:
            return "failed:dry_run_not_submitted"
        if not re.search(r"^CONFIRMATION_URL: https?://\S+", output, re.MULTILINE) or not re.search(
                r"^CONFIRMATION_MESSAGE: \S.+", output, re.MULTILINE):
            return "failed:submission_uncertain"
    return result


def run_codex(prompt: str, port: int, worker_dir: Path, log_path: Path,
              model: str | None, dry_run: bool, timeout: int, register, unregister) -> str:
    from applyagent.apply.chrome import _kill_process_tree
    output = worker_dir / "codex-result.txt"
    output.unlink(missing_ok=True)
    instructions = (
        "\nUse only the supplied browser MCP tools. If an action requires approval "
        "that cannot be obtained, stop; never bypass it. Do not send emails. "
        "End with exactly one RESULT code on its own line, without explanation on that line. "
        "For APPLIED also include CONFIRMATION_URL: <observed URL> and "
        "CONFIRMATION_MESSAGE: <exact observed success text>. Never invent confirmation."
    )
    proc = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(codex_command(port, output, model), stdin=subprocess.PIPE,
                                    stdout=log, stderr=subprocess.STDOUT, text=True,
                                    cwd=worker_dir, start_new_session=True)
            register(proc)
            try:
                proc.communicate(prompt + instructions, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid)
                proc.communicate()
                return "failed:submission_uncertain"
        if proc.returncode:
            return "failed:submission_uncertain"
        return parse_result(output.read_text() if output.exists() else "", dry_run)
    finally:
        unregister()
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc.pid)
            proc.wait()
