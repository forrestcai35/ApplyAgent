"""Bounded, noninteractive discovery runs for macOS/Linux schedulers."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from applyagent import config

log = logging.getLogger(__name__)


class _BoardErrors(logging.Handler):
    """JobSpy sometimes logs failed HTTP requests instead of raising."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record):
        self.count += 1


def validate_searches(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise TypeError("Search configuration must be a mapping")
    sources = cfg.get("discovery_sources", ["jobspy"])
    if not isinstance(sources, list) or not sources or any(s not in {"github", "public", "jobspy", "employer_api"} for s in sources):
        raise ValueError("discovery_sources must contain github, public, jobspy, or employer_api")
    if 'employer_api' in sources:
        from applyagent.discovery.employer_apis import endpoint
        for source in cfg.get('employer_boards', []):
            endpoint(source)
        for key in ('max_seconds', 'max_boards'):
            value = cfg.get('employer_api_settings', {}).get(key, 60 if key == 'max_seconds' else 15)
            if type(value) is not int or value < 1:
                raise ValueError(f'employer_api_settings.{key} must be a positive integer')
    quality = cfg.get("quality", {})
    for key in ("shortlist_limit", "min_score", 'max_per_company'):
        value = quality.get(key, {'shortlist_limit': 25, 'min_score': 60, 'max_per_company': 3}[key])
        if type(value) is not int or value < 1:
            raise ValueError(f"quality.{key} must be a positive integer")
    if "jobspy" not in sources:
        return
    for key, field in (("queries", "query"), ("locations", "location")):
        values = cfg.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Configure at least one {key} entry")
        if any(not isinstance(v, dict) or not isinstance(v.get(field), str) or not v[field].strip() for v in values):
            raise ValueError(f"Every {key} entry needs a nonempty {field}")
    boards = cfg.get("boards", cfg.get("sites", ["indeed", "linkedin", "zip_recruiter"]))
    supported = {"indeed", "linkedin", "glassdoor", "zip_recruiter", "google", "bayt", "naukri", "bdjobs"}
    if not isinstance(boards, list) or not boards or any(b not in supported for b in boards):
        raise ValueError("boards must be a nonempty list of supported JobSpy boards")
    for key in ("results_per_site", "hours_old"):
        value = cfg.get("defaults", {}).get(key, 72)
        if type(value) is not int or value < 1:
            raise ValueError(f"defaults.{key} must be a positive integer")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(report_path: Path) -> int:
    from applyagent.database import init_db
    from applyagent.discovery.jobspy import run_discovery

    config.load_env()
    cfg = config.load_search_config()
    validate_searches(cfg)
    conn = init_db()
    before = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM jobs").fetchone()[0]
    errors = _BoardErrors()
    board_loggers = [logger for name, logger in logging.Logger.manager.loggerDict.items()
                     if name.startswith("JobSpy:") and isinstance(logger, logging.Logger)]
    for logger in board_loggers:
        logger.addHandler(errors)
    stats = {"sources": {}, "errors": 0, "queries": 0}
    try:
        for source in cfg.get("discovery_sources", ["jobspy"]):
            try:
                if source == "github":
                    from applyagent.discovery.github_repos import run_github_repos
                    result = run_github_repos()
                    if not result.get("repos"):
                        result["errors"] = 1
                elif source == 'employer_api':
                    from applyagent.discovery.employer_apis import run_employer_apis
                    result = run_employer_apis(cfg)
                    if not result.get('sources'):
                        result['errors'] = 1
                elif source == "public":
                    from applyagent.discovery.public_sources import run_public_sources
                    result = run_public_sources(cfg)
                    if not result.get("sources"):
                        result["errors"] = 1
                else:
                    result = run_discovery(cfg)
                    if not result.get("queries"):
                        result["errors"] = 1
                stats["sources"][source] = result
                stats["errors"] += result.get("errors", 0)
                if result.get('backoff_skips') and not result.get('errors'):
                    stats['errors'] += 1  # Cached source outages still mean partial coverage.
                stats["queries"] += result.get("queries", 0)
            except Exception:
                log.exception("Discovery source failed: %s", source)
                stats["sources"][source] = {"errors": 1}
                stats["errors"] += 1
    finally:
        for logger in board_loggers:
            logger.removeHandler(errors)
    stats["board_error_logs"] = errors.count
    jobs = [dict(row) for row in conn.execute(
        "SELECT title, company, location, salary, site, strategy, description, full_description, url, application_url FROM jobs WHERE rowid > ?",
        (before,),
    )]
    collected = len(jobs)
    from applyagent.applications import filter_applications
    jobs = filter_applications(jobs)
    stats["suppressed_applied"] = collected - len(jobs)
    from applyagent.review import deferred_reason
    reviewed = [job for job in jobs if not deferred_reason(job)]
    stats['deferred_reviewed'] = len(jobs) - len(reviewed)
    jobs = reviewed
    if cfg.get("quality", {}).get("enabled"):
        from applyagent.discovery.quality import rank_jobs
        jobs = rank_jobs(jobs, limit=cfg["quality"].get("shortlist_limit", 25),
                         min_score=cfg["quality"].get("min_score", 60),
                         max_per_company=cfg['quality'].get('max_per_company', 3))
    stats["collected_new"] = collected
    stats["shortlisted_new"] = len(jobs)
    failed = bool(stats.get("errors") or errors.count)
    report = {
        "status": "partial" if failed else "ok",
        "finished_at": datetime.now(UTC).isoformat(),
        "stats": stats,
        "new_jobs": jobs,
        "note": "Shortlist scores reflect role/location evidence, not verified eligibility or employer quality. Check source errors in the log.",
    }
    _write_json(report_path, report)
    return 1 if failed else 0


def run_scheduled(timeout: int = 1800) -> int:
    """Run once under an OS lock. Exit 1 on failure, 2 on bad config, 124 on timeout."""
    if os.name != "posix":
        raise RuntimeError("Scheduled discovery currently requires macOS or Linux")
    import fcntl

    if timeout < 1:
        raise ValueError("timeout must be positive")
    # Never run unattended using the package's generic example preferences.
    if not config.SEARCH_CONFIG_PATH.is_file():
        log.error("Create %s before scheduling discovery", config.SEARCH_CONFIG_PATH)
        return 2
    try:
        cfg = config.load_search_config()
        validate_searches(cfg)
    except (ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        log.error("Invalid search configuration: %s", exc)
        return 2
    config.ensure_dirs()
    reports = config.APP_DIR / "reports"
    reports.mkdir(exist_ok=True)
    with (config.APP_DIR / "discovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("Discovery already running; skipped")
            return 0
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        report_path = reports / f"{stamp}.json"
        log_path = config.LOG_DIR / f"discovery-{stamp}.log"
        from applyagent.database import init_db

        conn = init_db()
        before = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM jobs").fetchone()[0]
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "applyagent.scheduled", str(report_path)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                # Worker inherits the lock so a supervisor crash cannot allow overlap.
                pass_fds=(lock.fileno(),),
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "status": "timeout" if code == 124 else "error",
                "new_jobs": [dict(row) for row in conn.execute(
                    "SELECT title, company, location, salary, site, strategy, description, full_description, url, application_url FROM jobs WHERE rowid > ?",
                    (before,),
                )],
                "note": "Run interrupted; includes jobs saved before the failure.",
            }
        if cfg.get("quality", {}).get("enabled") and "stats" not in report:
            from applyagent.applications import filter_applications
            from applyagent.discovery.quality import rank_jobs
            report["new_jobs"] = rank_jobs(filter_applications(report["new_jobs"]),
                                           limit=cfg["quality"].get("shortlist_limit", 25),
                                           min_score=cfg["quality"].get("min_score", 60))
        report.update(log=str(log_path), exit_code=code)
        _write_json(report_path, report)
        _write_json(reports / "latest.json", report)
        log.info("Discovery %s; report: %s", report["status"], report_path)
        return code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(_worker(Path(sys.argv[1])))
