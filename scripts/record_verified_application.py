"""Recover a verified submission receipt from an existing Codex browser log."""
import argparse
from pathlib import Path

from applyagent.apply.receipts import record_submission

parser = argparse.ArgumentParser()
parser.add_argument('job_url')
parser.add_argument('log', type=Path)
args = parser.parse_args()
print(record_submission(args.job_url, args.log))
