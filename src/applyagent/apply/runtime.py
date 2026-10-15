"""Shared run ownership and daily limits for autonomous application runs."""

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from zoneinfo import ZoneInfo

from applyagent import config
from applyagent.applications import application_identity


@contextmanager
def application_run_lock():
    config.APP_DIR.mkdir(parents=True, exist_ok=True)
    with (config.APP_DIR / "application.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another application run is active; no overlapping workers started") from error
        try:
            # A supervisor crash can release its OS lock while its agent still
            # runs. Never reset those claims or start competing browser workers.
            db = config.APP_DIR / "applyagent.db"
            if db.exists():
                with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
                    if conn.execute("SELECT 1 FROM jobs WHERE apply_status='in_progress' LIMIT 1").fetchone():
                        raise RuntimeError("Unresolved in-progress application; inspect saved logs before starting another run")
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def single_application_run(function):
    @wraps(function)
    def run(*args, **kwargs):
        with application_run_lock():
            return function(*args, **kwargs)
    return run


@dataclass(frozen=True)
class DailyBudget:
    date: str
    limit: int
    submitted: int
    timezone: str = "America/Los_Angeles"

    @property
    def remaining(self):
        return max(0, self.limit - self.submitted)


def daily_budget(now: datetime | None = None) -> DailyBudget:
    """Count durable receipts across runs, including DB successes awaiting receipts.

    A saved aggregate can conservatively reserve capacity, but never serves as
    evidence for an application receipt. Malformed state stops the run.
    """
    try:
        workflow = config.load_profile().get("application_workflow", {})
    except FileNotFoundError:
        workflow = {}
    limit = workflow.get("daily_submission_cap", 10)
    if type(limit) is not int or limit < 0:
        raise ValueError("daily_submission_cap must be a nonnegative integer")
    tz = ZoneInfo(workflow.get("timezone", "America/Los_Angeles"))
    day = (now or datetime.now(UTC)).astimezone(tz).date().isoformat()
    identities = []

    def count(entry):
        timestamp = entry.get("submitted_at")
        if not entry.get("submission_confirmed") or not timestamp:
            return
        if datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(tz).date().isoformat() != day:
            return
        urls = [entry.get("url"), entry.get("application_url"),
                *(e.get("url") for e in entry.get("evidence", []))]
        keys = {("url", application_identity(url)) for url in urls if url}
        if not keys:
            keys = {("role", (entry.get("company") or "").casefold(),
                     (entry.get("role") or "").casefold())}
        # Receipts connect a board URL, employer URL and confirmation URL. Merge
        # all aliases so receipt recovery cannot consume daily capacity twice.
        matches = [group for group in identities if group & keys]
        for group in matches:
            keys.update(group)
            identities.remove(group)
        identities.append(keys)

    registry = config.APP_DIR / "already_applied.json"
    if registry.exists():
        for entry in json.loads(registry.read_text())["applications"]:
            count(entry)
    reports = config.APP_DIR / "reports"
    for receipt in reports.glob(f"*-submitted-{day}*.json"):
        count(json.loads(receipt.read_text()))
    ledger = reports / f"applications-{day}.json"
    aggregate = 0
    if ledger.exists():
        saved = json.loads(ledger.read_text())
        for entry in saved.get("applications", []):
            count(entry)
        aggregate = saved.get("confirmed_submissions", 0)
        if type(aggregate) is not int or aggregate < 0:
            raise ValueError("Invalid confirmed_submissions in daily ledger")
    # Read only: do not create or initialize a database during this check.
    db = config.APP_DIR / "applyagent.db"
    if db.exists():
        with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                "SELECT company, title, url, application_url, applied_at FROM jobs "
                "WHERE apply_status='applied' AND applied_at IS NOT NULL"
            ):
                count(dict(row, role=row["title"], submitted_at=row["applied_at"], submission_confirmed=True))
    return DailyBudget(day, limit, max(aggregate, len(identities)), tz.key)
