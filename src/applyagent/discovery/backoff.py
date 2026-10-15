"""Respect board failures across queries and scheduled runs."""

import json
import logging
from datetime import UTC, datetime, timedelta

from applyagent import config


def failure_kind(message: str) -> str | None:
    text = message.lower()
    if any(token in text for token in ('nameresolutionerror', 'could not resolve', 'failed to resolve', 'nodename nor servname')):
        return 'network_unavailable'
    if any(token in text for token in ('429', '403', '401', 'captcha', 'no jobkeysmap')):
        return 'restricted_or_unavailable'
    return None


class BoardBackoff(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.path = config.APP_DIR / 'discovery-backoff.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.network_unavailable = False
        self.skipped = 0
        self.failures = 0

    def emit(self, record):
        site = record.name.split(':')[-1].lower()
        site = {'ziprecruiter': 'zip_recruiter'}.get(site, site)
        self.failed(site, record.getMessage())

    def failed(self, site, message):
        kind = failure_kind(message)
        if not kind:
            return
        self.failures += 1
        if kind == 'network_unavailable':
            # A sandbox DNS failure must not poison a later network-capable run.
            self.network_unavailable = True
            return
        self.state[site] = {'reason': kind, 'retry_after':
                            (datetime.now(UTC) + timedelta(hours=24)).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def allows(self, site):
        saved = self.state.get(site)
        allowed = not self.network_unavailable and (not saved or
            datetime.fromisoformat(saved['retry_after']) <= datetime.now(UTC))
        self.skipped += int(not allowed)
        return allowed
