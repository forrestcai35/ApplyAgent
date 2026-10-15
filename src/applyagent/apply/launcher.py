"""Apply orchestration: acquire jobs, run agent sessions, track results.

This is the main entry point for the apply pipeline. It pulls jobs from
the database, launches Chrome + an AI agent for each one, parses the
result, and updates the database. Supports parallel workers via --workers.

Two agent backends are available:
  - Claude Code (default): spawns `claude` CLI with Playwright MCP
  - Codex: bounded codex exec with Playwright MCP and final-message verification
"""

import atexit
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.live import Live

from applyagent import config
from applyagent.database import get_connection
from applyagent.apply import chrome, dashboard, prompt as prompt_mod
from applyagent.apply.runtime import daily_budget, single_application_run
from applyagent.apply.chrome import (
    launch_chrome, cleanup_worker, kill_all_chrome,
    reset_worker_dir, cleanup_on_exit, _kill_process_tree,
    BASE_CDP_PORT,
)
from applyagent.apply.dashboard import (
    init_worker, update_state, add_event, get_state,
    render_full, get_totals,
)

logger = logging.getLogger(__name__)

# Blocked sites loaded from config/sites.yaml
def _load_blocked():
    from applyagent.config import load_blocked_sites
    return load_blocked_sites()

# How often to poll the DB when the queue is empty (seconds)
POLL_INTERVAL = config.DEFAULTS["poll_interval"]

# Thread-safe shutdown coordination
_stop_event = threading.Event()

# Track active Claude Code processes for skip (Ctrl+C) handling
_claude_procs: dict[int, subprocess.Popen] = {}
_claude_lock = threading.Lock()

# Ctrl+C counter: list wrapper so signal handler and worker loop can share it
_ctrl_c_state = {"count": 0}

# Register cleanup on exit
atexit.register(cleanup_on_exit)
if platform.system() != "Windows":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# MCP config
# ---------------------------------------------------------------------------

def _make_mcp_config(cdp_port: int) -> dict:
    """Build MCP config dict for a specific CDP port."""
    return {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest",
                    f"--cdp-endpoint=http://localhost:{cdp_port}",
                    f"--viewport-size={config.DEFAULTS['viewport']}",
                ],
            },
            "gmail": {
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
            },
        }
    }


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def acquire_job(target_url: str | None = None, min_score: int = 7,
                worker_id: int = 0) -> dict | None:
    """Atomically acquire the next job to apply to.

    Args:
        target_url: Apply to a specific URL instead of picking from queue.
        min_score: Minimum fit_score threshold.
        worker_id: Worker claiming this job (for tracking).

    Returns:
        Job dict or None if the queue is empty.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")

        if target_url:
            from applyagent.applications import application_identity
            target = application_identity(target_url)
            candidates = conn.execute("""
                SELECT url, title, company, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE tailored_resume_path IS NOT NULL
                  AND applied_at IS NULL
                  AND (apply_status IS NULL OR apply_status = 'failed')
                  AND COALESCE(apply_attempts, 0) < ?
            """, (config.DEFAULTS["max_apply_attempts"],)).fetchall()
            row = next((candidate for candidate in candidates
                        if any(application_identity(value) == target for value in
                               (candidate["url"], candidate["application_url"]) if value)), None)

        else:
            blocked_sites, blocked_patterns = _load_blocked()
            # Build parameterized filters to avoid SQL injection
            params: list = [min_score]
            site_clause = ""
            if blocked_sites:
                placeholders = ",".join("?" * len(blocked_sites))
                site_clause = f"AND site NOT IN ({placeholders})"
                params.extend(blocked_sites)
            url_clauses = ""
            if blocked_patterns:
                url_clauses = " ".join(f"AND url NOT LIKE ?" for _ in blocked_patterns)
                params.extend(blocked_patterns)
            row = conn.execute(f"""
                SELECT url, title, company, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE tailored_resume_path IS NOT NULL
                  AND duplicate_of IS NULL
                  AND applied_at IS NULL
                  AND (apply_status IS NULL OR apply_status = 'failed')
                  AND (apply_attempts IS NULL OR apply_attempts < ?)
                  AND fit_score >= ?
                  {site_clause}
                  {url_clauses}
                ORDER BY fit_score DESC, url
            """, [config.DEFAULTS["max_apply_attempts"]] + params).fetchall()
            from applyagent.applications import filter_applications
            eligible = filter_applications([dict(candidate) for candidate in row])
            from applyagent.review import deferred_reason
            eligible = [candidate for candidate in eligible if not deferred_reason(candidate)]
            from applyagent.applications import application_identity
            attempted_targets: dict[str, set[str]] = {}
            for previous in conn.execute(
                "SELECT url, application_url FROM jobs WHERE applied_at IS NOT NULL "
                "OR apply_status IN ('applied', 'in_progress', 'failed', 'skipped', 'manual')"
            ):
                key = application_identity(previous["application_url"] or previous["url"])
                attempted_targets.setdefault(key, set()).add(previous["url"])
            eligible = [candidate for candidate in eligible if not (
                attempted_targets.get(application_identity(candidate["application_url"] or candidate["url"]), set())
                - {candidate["url"]}
            )]
            row = eligible[0] if eligible else None

        from applyagent.applications import filter_applications
        if row and not filter_applications([dict(row)]):
            row = None
        if row:
            from applyagent.review import deferred_reason
            if deferred_reason(dict(row)):
                row = None
        if row and target_url:
            identity = application_identity(row["application_url"] or row["url"])
            if any(application_identity(previous["application_url"] or previous["url"]) == identity
                   for previous in conn.execute(
                       "SELECT url, application_url FROM jobs WHERE url != ? AND "
                       "(applied_at IS NOT NULL OR apply_status IN ('applied', 'in_progress', 'failed', 'skipped', 'manual'))",
                       (row['url'],))):
                row = None
        if not row:
            conn.rollback()
            return None

        # Skip manual ATS sites (unsolvable CAPTCHAs)
        from applyagent.config import is_manual_ats
        apply_url = row["application_url"] or row["url"]
        if is_manual_ats(apply_url):
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            logger.info("Skipping manual ATS: %s", row["url"][:80])
            if target_url:
                return None
            return acquire_job(min_score=min_score, worker_id=worker_id)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
        """, (f"worker-{worker_id}", now, row["url"]))
        conn.commit()

        return dict(row)
    except Exception:
        conn.rollback()
        raise


def mark_result(url: str, status: str, error: str | None = None,
                permanent: bool = False, duration_ms: int | None = None,
                task_id: str | None = None) -> None:
    """Update a job's apply status in the database."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute("""
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """, (now, duration_ms, task_id, url))
    else:
        attempts = 99 if permanent else "COALESCE(apply_attempts, 0) + 1"
        conn.execute(f"""
            UPDATE jobs SET apply_status = ?, apply_error = ?,
                           apply_attempts = {attempts}, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """, (status, error or "unknown", duration_ms, task_id, url))
    conn.commit()


def release_lock(url: str) -> None:
    """Release the in_progress lock without changing status."""
    conn = get_connection()
    conn.execute(
        "UPDATE jobs SET apply_status = NULL, agent_id = NULL WHERE url = ? AND apply_status = 'in_progress'",
        (url,),
    )
    conn.commit()


def mark_skipped(url: str) -> None:
    """Mark a job as user-skipped so it won't be re-acquired this session."""
    conn = get_connection()
    conn.execute(
        "UPDATE jobs SET apply_status = 'skipped', agent_id = NULL WHERE url = ?",
        (url,),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Utility modes (--gen, --mark-applied, --mark-failed, --reset-failed)
# ---------------------------------------------------------------------------

def gen_prompt(target_url: str, min_score: int = 7,
               model: str = "sonnet", worker_id: int = 0) -> Path | None:
    """Generate a prompt file and print the Claude CLI command for manual debugging.

    Returns:
        Path to the generated prompt file, or None if no job found.
    """
    job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
    if not job:
        return None

    # Read resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    prompt = prompt_mod.build_prompt(job=job, tailored_resume=resume_text)

    # Release the lock so the job stays available
    release_lock(job["url"])

    # Write prompt file
    config.ensure_dirs()
    site_slug = (job.get("site") or "unknown")[:20].replace(" ", "_")
    prompt_file = config.LOG_DIR / f"prompt_{site_slug}_{job['title'][:30].replace(' ', '_')}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    # Write MCP config for reference
    port = BASE_CDP_PORT + worker_id
    mcp_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    return prompt_file


def mark_job(url: str, status: str, reason: str | None = None) -> None:
    """Manually mark a job's apply status in the database.

    Args:
        url: Job URL to mark.
        status: Either 'applied' or 'failed'.
        reason: Failure reason (only for status='failed').
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute("""
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL
            WHERE url = ?
        """, (now, url))
    else:
        conn.execute("""
            UPDATE jobs SET apply_status = 'failed', apply_error = ?,
                           apply_attempts = 99, agent_id = NULL
            WHERE url = ?
        """, (reason or "manual", url))
    conn.commit()


def reset_failed() -> int:
    """Reset transient failures only; preserve uncertain submissions and holds.

    Returns:
        Number of jobs reset.
    """
    conn = get_connection()
    excluded = sorted(PERMANENT_FAILURES)
    placeholders = ','.join('?' for _ in excluded)
    prefixes = ' '.join("AND COALESCE(apply_error, '') NOT LIKE ?" for _ in PERMANENT_PREFIXES)
    parameters = excluded + [prefix + '%' for prefix in PERMANENT_PREFIXES]
    cursor = conn.execute(f"""
        UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                       apply_attempts = 0, agent_id = NULL
        WHERE apply_status = 'failed'
          AND COALESCE(apply_error, '') NOT IN ({placeholders})
          {prefixes}
    """, parameters)
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------

def run_job(job: dict, port: int, worker_id: int = 0,
            model: str | None = None, dry_run: bool = False, backend: str | None = None) -> tuple[str, int]:
    """Spawn a Claude Code session for one job application.

    Returns:
        Tuple of (status_string, duration_ms). Status is one of:
        'applied', 'expired', 'captcha', 'login_issue',
        'failed:reason', or 'skipped'.
    """
    from applyagent.apply.backend import selected_backend, run_codex
    apply_timeout = int(os.environ.get('APPLYAGENT_APPLY_TIMEOUT', '600'))
    if job.get('_run_deadline') is not None:
        apply_timeout = min(apply_timeout, max(1, int(job['_run_deadline'] - time.monotonic())))
    backend = selected_backend(backend)
    if backend == "codex":
        worker_dir = reset_worker_dir(worker_id)
        config.ensure_dirs()
        resume = job.get("tailored_resume_path")
        txt = Path(resume).with_suffix(".txt") if resume else None
        agent_prompt = prompt_mod.build_prompt(
            job=job, tailored_resume=txt.read_text() if txt and txt.exists() else "",
            dry_run=dry_run, upload_dir=worker_dir)
        def register(proc):
            with _claude_lock:
                _claude_procs[worker_id] = proc
        def unregister():
            with _claude_lock:
                _claude_procs.pop(worker_id, None)
        start = time.monotonic()
        update_state(worker_id, status="applying", job_title=job["title"], start_time=time.time())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        job['_confirmation_log'] = config.LOG_DIR / f"codex_{stamp}_w{worker_id}.jsonl"
        result = run_codex(agent_prompt, port, worker_dir,
                           job['_confirmation_log'],
                           model, dry_run, apply_timeout,
                           register, unregister)
        add_event(f"[W{worker_id}] Codex: {result}")
        return result, int((time.monotonic() - start) * 1000)
    model = model or "sonnet"
    # Read tailored resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    # Build the prompt
    agent_prompt = prompt_mod.build_prompt(
        job=job,
        tailored_resume=resume_text,
        dry_run=dry_run,
    )

    # Write per-worker MCP config
    mcp_config_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_config_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    # Build claude command
    cmd = [
        "claude",
        "--model", model,
        "-p",
        "--mcp-config", str(mcp_config_path),
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--disallowedTools", (
            "mcp__gmail__draft_email,mcp__gmail__modify_email,"
            "mcp__gmail__delete_email,mcp__gmail__download_attachment,"
            "mcp__gmail__batch_modify_emails,mcp__gmail__batch_delete_emails,"
            "mcp__gmail__create_label,mcp__gmail__update_label,"
            "mcp__gmail__delete_label,mcp__gmail__get_or_create_label,"
            "mcp__gmail__list_email_labels,mcp__gmail__create_filter,"
            "mcp__gmail__list_filters,mcp__gmail__get_filter,"
            "mcp__gmail__delete_filter"
        ),
        "--output-format", "stream-json",
        "--verbose", "-",
    ]

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    worker_dir = reset_worker_dir(worker_id)

    update_state(worker_id, status="applying", job_title=job["title"],
                 company=job.get("site", ""), score=job.get("fit_score", 0),
                 start_time=time.time(), actions=0, last_action="starting")
    add_event(f"[W{worker_id}] Starting: {job['title'][:40]} @ {job.get('site', '')}")

    worker_log = config.LOG_DIR / f"worker-{worker_id}.log"
    ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = (
        f"\n{'=' * 60}\n"
        f"[{ts_header}] {job['title']} @ {job.get('site', '')}\n"
        f"URL: {job.get('application_url') or job['url']}\n"
        f"Score: {job.get('fit_score', 'N/A')}/10\n"
        f"{'=' * 60}\n"
    )

    start = time.time()
    stats: dict = {}
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(worker_dir),
            start_new_session=True,
        )
        with _claude_lock:
            _claude_procs[worker_id] = proc

        proc.stdin.write(agent_prompt)
        proc.stdin.close()

        text_parts: list[str] = []
        with open(worker_log, "a", encoding="utf-8") as lf:
            lf.write(log_header)

            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_type = msg.get("type")
                    if msg_type == "assistant":
                        for block in msg.get("message", {}).get("content", []):
                            bt = block.get("type")
                            if bt == "text":
                                text_parts.append(block["text"])
                                lf.write(block["text"] + "\n")
                            elif bt == "tool_use":
                                name = (
                                    block.get("name", "")
                                    .replace("mcp__playwright__", "")
                                    .replace("mcp__gmail__", "gmail:")
                                )
                                inp = block.get("input", {})
                                if "url" in inp:
                                    desc = f"{name} {inp['url'][:60]}"
                                elif "ref" in inp:
                                    desc = f"{name} {inp.get('element', inp.get('text', ''))}"[:50]
                                elif "fields" in inp:
                                    desc = f"{name} ({len(inp['fields'])} fields)"
                                elif "paths" in inp:
                                    desc = f"{name} upload"
                                else:
                                    desc = name

                                lf.write(f"  >> {desc}\n")
                                ws = get_state(worker_id)
                                cur_actions = ws.actions if ws else 0
                                update_state(worker_id,
                                             actions=cur_actions + 1,
                                             last_action=desc[:35])
                    elif msg_type == "result":
                        stats = {
                            "input_tokens": msg.get("usage", {}).get("input_tokens", 0),
                            "output_tokens": msg.get("usage", {}).get("output_tokens", 0),
                            "cache_read": msg.get("usage", {}).get("cache_read_input_tokens", 0),
                            "cache_create": msg.get("usage", {}).get("cache_creation_input_tokens", 0),
                            "cost_usd": msg.get("total_cost_usd", 0),
                            "turns": msg.get("num_turns", 0),
                        }
                        text_parts.append(msg.get("result", ""))
                except json.JSONDecodeError:
                    text_parts.append(line)
                    lf.write(line + "\n")

        proc.wait(timeout=min(300, apply_timeout))
        returncode = proc.returncode
        proc = None

        if returncode and returncode < 0:
            # Only treat as a user-initiated skip if Ctrl+C was pressed recently
            # (within last 5 seconds). Otherwise it's a process crash.
            if time.time() - _ctrl_c_state.get("last_time", 0) < 5.0:
                return "skipped", int((time.time() - start) * 1000)
            duration_ms = int((time.time() - start) * 1000)
            add_event(f"[W{worker_id}] CRASHED (signal {-returncode}, {duration_ms // 1000}s)")
            update_state(worker_id, status="failed",
                         last_action=f"crashed (signal {-returncode})")
            return f"failed:process_crashed", duration_ms

        output = "\n".join(text_parts)
        elapsed = int(time.time() - start)
        duration_ms = int((time.time() - start) * 1000)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_log = config.LOG_DIR / f"claude_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}.txt"
        job_log.write_text(output, encoding="utf-8")

        if stats:
            cost = stats.get("cost_usd", 0)
            ws = get_state(worker_id)
            prev_cost = ws.total_cost if ws else 0.0
            update_state(worker_id, total_cost=prev_cost + cost)

        def _clean_reason(s: str) -> str:
            return re.sub(r'[*`"]+$', '', s).strip()

        for result_status in ["APPLIED", "EXPIRED", "CAPTCHA", "LOGIN_ISSUE"]:
            if f"RESULT:{result_status}" in output:
                add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                update_state(worker_id, status=result_status.lower(),
                             last_action=f"{result_status} ({elapsed}s)")
                return result_status.lower(), duration_ms

        if "RESULT:FAILED" in output:
            for out_line in output.split("\n"):
                if "RESULT:FAILED" in out_line:
                    reason = (
                        out_line.split("RESULT:FAILED:")[-1].strip()
                        if ":" in out_line[out_line.index("FAILED") + 6:]
                        else "unknown"
                    )
                    reason = _clean_reason(reason)
                    PROMOTE_TO_STATUS = {"captcha", "expired", "login_issue"}
                    if reason in PROMOTE_TO_STATUS:
                        add_event(f"[W{worker_id}] {reason.upper()} ({elapsed}s): {job['title'][:30]}")
                        update_state(worker_id, status=reason,
                                     last_action=f"{reason.upper()} ({elapsed}s)")
                        return reason, duration_ms
                    add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
                    update_state(worker_id, status="failed",
                                 last_action=f"FAILED: {reason[:25]}")
                    return f"failed:{reason}", duration_ms
            return "failed:unknown", duration_ms

        add_event(f"[W{worker_id}] NO RESULT ({elapsed}s)")
        update_state(worker_id, status="failed", last_action=f"no result ({elapsed}s)")
        # If Claude exited in under 15s with no result, the job is almost
        # certainly broken (bad URL, prompt parse error, etc.) -- don't retry.
        if elapsed < 15:
            return "failed:instant_failure", duration_ms
        return "failed:no_result_line", duration_ms

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        elapsed = int(time.time() - start)
        add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s)")
        update_state(worker_id, status="failed", last_action=f"TIMEOUT ({elapsed}s)")
        return "failed:timeout", duration_ms
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        add_event(f"[W{worker_id}] ERROR: {str(e)[:40]}")
        update_state(worker_id, status="failed", last_action=f"ERROR: {str(e)[:25]}")
        return f"failed:{str(e)[:100]}", duration_ms
    finally:
        with _claude_lock:
            _claude_procs.pop(worker_id, None)
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc.pid)


# ---------------------------------------------------------------------------
# Permanent failure classification
# ---------------------------------------------------------------------------

PERMANENT_FAILURES: set[str] = {
    "expired", "captcha", "login_issue",
    "not_eligible_location", "not_eligible_salary",
    "already_applied", "account_required",
    "not_a_job_application", "unsafe_permissions",
    "unsafe_verification", "sso_required",
    "site_blocked", "cloudflare_blocked", "blocked_by_cloudflare",
    "instant_failure", "submission_uncertain", "approval_required", "unknown_required_answer", "page_error", "form_save_failed",
    "excluded_required_field",
}

PERMANENT_PREFIXES: tuple[str, ...] = ("site_blocked", "cloudflare", "blocked_by")


def _is_permanent_failure(result: str) -> bool:
    """Determine if a failure should never be retried."""
    reason = result.split(":", 1)[-1] if ":" in result else result
    return (
        result in PERMANENT_FAILURES
        or reason in PERMANENT_FAILURES
        or any(reason.startswith(p) for p in PERMANENT_PREFIXES)
    )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(worker_id: int = 0, limit: int = 1,
                target_url: str | None = None,
                min_score: int = 7, headless: bool = False,
                model: str | None = None, dry_run: bool = False, backend: str | None = None,
                until_submitted: int = 0, refill: bool = True,
                refill_rounds: int = 3) -> tuple[int, int]:
    """Run jobs sequentially until limit is reached or queue is empty.

    Args:
        worker_id: Numeric worker identifier.
        limit: Max jobs to process (0 = continuous).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome headless.
        model: Claude model name.
        dry_run: Don't click Submit.

    Returns:
        Tuple of (applied_count, failed_count).
    """
    if until_submitted and (dry_run or target_url):
        raise ValueError("Submission targets require live queue mode")
    if refill_rounds < 0:
        raise ValueError("refill_rounds must be nonnegative")
    applied = 0
    failed = 0
    continuous = limit == 0
    jobs_done = 0
    empty_polls = 0
    rapid_fails = 0
    refills = 0
    try:
        workflow = config.load_profile().get('application_workflow', {})
    except FileNotFoundError:
        workflow = {}
    run_seconds = workflow.get('max_run_seconds', 900)
    max_attempts = workflow.get('max_attempts_per_run', 15)
    if any(type(value) is not int or value < 1 for value in (run_seconds, max_attempts)):
        raise ValueError('Run time and attempt limits must be positive integers')
    deadline = time.monotonic() + run_seconds
    attempts = 0
    day_baselines = {}
    day_applied = {}
    port = BASE_CDP_PORT + worker_id

    RAPID_FAIL_THRESHOLD_MS = 15_000
    MAX_RAPID_FAILS = 5
    RAPID_FAIL_BASE_DELAY = 30

    while not _stop_event.is_set():
        if time.monotonic() >= deadline or attempts >= max_attempts:
            add_event(f'[W{worker_id}] Run budget reached; {applied} confirmed, {attempts} attempts')
            break
        if until_submitted:
            if applied >= until_submitted:
                break
        elif not continuous and jobs_done >= limit:
            break

        if not dry_run:
            budget = daily_budget()
            day_baselines.setdefault(budget.date, budget.submitted)
            remaining = min(budget.remaining, max(0, budget.limit - (
                day_baselines[budget.date] + day_applied.get(budget.date, 0))))
            if not remaining:
                add_event(f"[W{worker_id}] Daily submission cap reached for {budget.date}")
                break

        update_state(worker_id, status="idle", job_title="", company="",
                     last_action="waiting for job", actions=0)

        job = acquire_job(target_url=target_url, min_score=min_score,
                          worker_id=worker_id)
        if not job:
            if refill and not target_url and refills < refill_rounds:
                from applyagent.apply.refill import replenish
                refills += 1
                add_event(f"[W{worker_id}] Replenishing prospects (round {refills}/{refill_rounds})")
                try:
                    prepared = replenish(min_score=min_score, discover=refills == 1,
                                         max_candidates=workflow.get('max_refill_candidates', 12),
                                         deadline=deadline)
                except Exception as error:
                    logger.exception("Prospect replenishment failed")
                    update_state(worker_id, status='error', last_action='preparation failed')
                    raise RuntimeError('Application preparation failed; see private scoring diagnostics') from error
                if prepared:
                    continue
                add_event(f"[W{worker_id}] No additional suitable prospects prepared")
            if until_submitted or (refill and not target_url):
                add_event(f"[W{worker_id}] Queue exhausted; {applied} confirmed, target not guaranteed")
                break
            if not continuous or target_url:
                add_event(f"[W{worker_id}] Queue empty")
                update_state(worker_id, status="done", last_action="queue empty")
                break
            empty_polls += 1
            update_state(worker_id, status="idle",
                         last_action=f"polling ({empty_polls})")
            if empty_polls == 1:
                add_event(f"[W{worker_id}] Queue empty, polling every {POLL_INTERVAL}s...")
            if _stop_event.wait(timeout=POLL_INTERVAL):
                break
            continue

        empty_polls = 0
        attempts += 1
        job['_run_deadline'] = deadline

        chrome_proc = None
        try:
            add_event(f"[W{worker_id}] Launching Chrome...")
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless)

            result, duration_ms = run_job(
                job, port=port, worker_id=worker_id,
                model=model, dry_run=dry_run, backend=backend)

            if result == "skipped":
                mark_skipped(job["url"])
                add_event(f"[W{worker_id}] Skipped: {job['title'][:30]}")
                continue
            elif result == "applied":
                mark_result(job["url"], "applied", duration_ms=duration_ms)
                applied += 1
                if not dry_run:
                    completed_day = datetime.now(timezone.utc).astimezone(ZoneInfo(budget.timezone)).date().isoformat()
                    day_baselines.setdefault(completed_day, 0)
                    day_applied[completed_day] = day_applied.get(completed_day, 0) + 1
                if job.get('_confirmation_log'):
                    from applyagent.apply.receipts import record_submission
                    try:
                        record_submission(job['url'], job['_confirmation_log'])
                    except Exception:
                        # Keep the applied DB record: a bookkeeping failure must never cause reapplication.
                        logger.exception('Receipt persistence failed for %s; recover from saved log', job['url'])
                        add_event(f"[W{worker_id}] Submitted; receipt needs recovery from saved log")
                rapid_fails = 0
                update_state(worker_id, jobs_applied=applied,
                             jobs_done=applied + failed)
            else:
                is_rapid = not until_submitted and duration_ms < RAPID_FAIL_THRESHOLD_MS and not _is_permanent_failure(result)
                if is_rapid:
                    rapid_fails += 1
                    release_lock(job["url"])
                    if rapid_fails >= MAX_RAPID_FAILS:
                        add_event(
                            f"[W{worker_id}] {rapid_fails} consecutive instant failures — "
                            f"Claude Code likely rate-limited. Stopping.")
                        update_state(worker_id, status="rate_limited",
                                     last_action="rate limited, stopping")
                        break
                    delay = RAPID_FAIL_BASE_DELAY * (2 ** (rapid_fails - 1))
                    add_event(
                        f"[W{worker_id}] Instant failure ({duration_ms // 1000}s) — "
                        f"likely rate-limited. Waiting {delay}s before retry "
                        f"({rapid_fails}/{MAX_RAPID_FAILS})...")
                    update_state(worker_id, status="backoff",
                                 last_action=f"rate-limit backoff {delay}s")
                    if _stop_event.wait(timeout=delay):
                        break
                    continue
                else:
                    rapid_fails = 0
                    reason = result.split(":", 1)[-1] if ":" in result else result
                    mark_result(job["url"], "failed", reason,
                                permanent=bool(until_submitted) or _is_permanent_failure(result),
                                duration_ms=duration_ms)
                    failed += 1
                    update_state(worker_id, jobs_failed=failed,
                                 jobs_done=applied + failed)

        except KeyboardInterrupt:
            if _stop_event.is_set():
                release_lock(job["url"])
                break
            mark_skipped(job["url"])
            add_event(f"[W{worker_id}] Job skipped (Ctrl+C)")
            continue
        except Exception as e:
            logger.exception("Worker %d launcher error", worker_id)
            add_event(f"[W{worker_id}] Launcher error: {str(e)[:40]}")
            mark_result(job["url"], "failed", "launcher_error", permanent=bool(until_submitted))
            failed += 1
            update_state(worker_id, jobs_failed=failed)
        finally:
            if chrome_proc:
                cleanup_worker(worker_id, chrome_proc)

        jobs_done += 1
        if target_url:
            break

    update_state(worker_id, status="done", last_action="finished")
    return applied, failed


# ---------------------------------------------------------------------------
# Main entry point (called from cli.py)
# ---------------------------------------------------------------------------

@single_application_run
def main(limit: int = 1, target_url: str | None = None,
         min_score: int = 7, headless: bool = False, model: str | None = None,
         dry_run: bool = False, continuous: bool = False,
         poll_interval: int = 60, workers: int = 1, backend: str | None = None,
         until_submitted: int = 0, refill: bool = True,
         refill_rounds: int = 3) -> None:
    """Launch the apply pipeline.

    Args:
        limit: Max jobs to apply to (0 or with continuous=True means run forever).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome in headless mode.
        model: Claude model name.
        dry_run: Don't click Submit.
        continuous: Run forever, polling for new jobs.
        poll_interval: Seconds between DB polls when queue is empty.
        workers: Number of parallel workers (default 1).
    """
    if until_submitted and (workers != 1 or dry_run or target_url):
        raise ValueError("Submission targets require one live queue worker")
    if not dry_run and workers != 1:
        raise ValueError("Live applications require one worker to enforce the shared daily cap")
    global POLL_INTERVAL
    POLL_INTERVAL = poll_interval
    _stop_event.clear()

    config.ensure_dirs()
    console = Console()

    if until_submitted:
        effective_limit = 0
        mode_label = f"{until_submitted} confirmed submissions"
    elif continuous:
        effective_limit = 0
        mode_label = "continuous"
    else:
        effective_limit = limit
        mode_label = f"{limit} jobs"

    # Initialize dashboard for all workers
    for i in range(workers):
        init_worker(i)

    worker_label = f"{workers} worker{'s' if workers > 1 else ''}"
    console.print(f"Launching apply pipeline ({mode_label}, {worker_label}, poll every {POLL_INTERVAL}s)...")
    console.print("[dim]Ctrl+C = skip current job(s) | Ctrl+C x2 = stop[/dim]")

    # Double Ctrl+C handler: first press skips, rapid second press stops.
    # Uses module-level _ctrl_c_state so worker_loop can reset after each job.
    _ctrl_c_state["count"] = 0
    _ctrl_c_state["last_time"] = 0.0

    def _sigint_handler(sig, frame):
        now = time.time()
        # Treat as a "new first press" if >3s since last Ctrl+C
        if now - _ctrl_c_state["last_time"] > 3.0:
            _ctrl_c_state["count"] = 0
        _ctrl_c_state["count"] += 1
        _ctrl_c_state["last_time"] = now

        if _ctrl_c_state["count"] == 1:
            console.print("\n[yellow]Skipping current job(s)... (Ctrl+C again within 3s to STOP)[/yellow]")
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
        else:
            console.print("\n[red bold]STOPPING[/red bold]")
            _stop_event.set()
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
            kill_all_chrome()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(render_full(), console=console, refresh_per_second=2) as live:
            # Daemon thread for display refresh only (no business logic)
            _dashboard_running = True

            def _refresh():
                while _dashboard_running:
                    live.update(render_full())
                    time.sleep(0.5)

            refresh_thread = threading.Thread(target=_refresh, daemon=True)
            refresh_thread.start()

            if workers == 1:
                # Single worker — run directly in main thread
                total_applied, total_failed = worker_loop(
                    worker_id=0,
                    limit=effective_limit,
                    target_url=target_url,
                    min_score=min_score,
                    headless=headless,
                    model=model,
                    dry_run=dry_run, backend=backend, until_submitted=until_submitted,
                    refill=refill, refill_rounds=refill_rounds,
                )
            else:
                # Multi-worker — distribute limit across workers
                if effective_limit:
                    base = effective_limit // workers
                    extra = effective_limit % workers
                    limits = [base + (1 if i < extra else 0)
                              for i in range(workers)]
                else:
                    limits = [0] * workers  # continuous mode

                with ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="apply-worker") as executor:
                    futures = {
                        executor.submit(
                            worker_loop,
                            worker_id=i,
                            limit=limits[i],
                            target_url=target_url,
                            min_score=min_score,
                            headless=headless,
                            model=model,
                            dry_run=dry_run, backend=backend,
                            refill=False,
                        ): i
                        for i in range(workers)
                    }

                    results: list[tuple[int, int]] = []
                    for future in as_completed(futures):
                        wid = futures[future]
                        try:
                            results.append(future.result())
                        except Exception:
                            logger.exception("Worker %d crashed", wid)
                            results.append((0, 0))

                total_applied = sum(r[0] for r in results)
                total_failed = sum(r[1] for r in results)

            _dashboard_running = False
            refresh_thread.join(timeout=2)
            live.update(render_full())

        totals = get_totals()
        console.print(
            f"\n[bold]Run ended: {total_applied} applied, {total_failed} failed "
            f"(${totals['cost']:.3f})[/bold]"
        )
        console.print(f"Logs: {config.LOG_DIR}")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        kill_all_chrome()
