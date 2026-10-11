# Scheduled job discovery

`applyagent scheduled-discover` runs configured sources once, without interactive prompts, resume generation, or application submission. Sources can include curated GitHub job tables, RSS feeds, public HTML JobPosting data, and JobSpy boards. Existing Workday and AI-powered direct-site discovery remain available through `applyagent run discover`. The scheduled collectors use HTTP without a browser or paid scraping API.

## Install

Requires Python 3.11+ on macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[discovery]'
mkdir -p "$HOME/.applyagent"
```

The discovery extra installs JobSpy 1.2 or newer, which no longer needs the legacy `--no-deps` installation workaround.

## Configure your search

Create `~/.applyagent/searches.yaml`. These are example preferences; replace them with your own before running:

```yaml
queries:
  - query: backend engineer
    tier: 1
locations:
  - location: San Francisco, CA
    remote: false
  - location: United States
    remote: true
boards: [linkedin, indeed, zip_recruiter]
country: USA
defaults:
  results_per_site: 25
  hours_old: 72
exclude_titles: [intern, director, principal]
# Optional substring filters; omit to trust the location passed to the board.
# Reject patterns take precedence, including for remote jobs.
location:
  accept_patterns: []
  reject_patterns: []
  allow_remote: false  # true accepts jobs explicitly marked remote by the board
  accept_unknown: true  # false rejects jobs with unknown location
```

A missing search file stops the unattended command instead of using the package example. `APPLYAGENT_DIR` can point to a different persistent data directory; set it consistently for manual and scheduled runs. No API key or LinkedIn account credentials are needed for public discovery.

## Verify once

```sh
.venv/bin/applyagent scheduled-discover --timeout 1800
```

Output lives under the data directory:

- `applyagent.db`: persistent jobs, deduplicated by listing URL across runs.
- `reports/latest.json`: latest run status, counts, new roles and source/application URLs.
- `reports/<UTC timestamp>.json`: per-run history.
- `logs/discovery-<UTC timestamp>.log`: scraper output and errors.

This does not merge the same vacancy advertised under different board URLs. The existing `applyagent run dedup` command handles cross-source duplicates separately. Optional AI scoring uses the existing profile/resume setup and `applyagent run score` after enrichment.

## Cron on an always-on host

After verifying your preferences and the first run, use `crontab -e` and add the following, replacing both paths with absolute paths on your host:

```cron
0 8 * * * APPLYAGENT_DIR=/absolute/path/to/data /absolute/path/to/ApplyAgent/.venv/bin/applyagent scheduled-discover --timeout 1800
```

This runs at 8 a.m. in the cron host's timezone. Set the host timezone to `America/Los_Angeles` for Pacific time with daylight-saving changes. Cron won't run while your laptop is asleep; use an always-on Mac/Linux machine for continuous operation. Keep the data directory persistent across deployments. Remove the crontab entry to stop recurring runs.

The command uses an OS lock, skips overlapping scheduled runs, and terminates a run exceeding its timeout. The lock also survives a supervisor crash while its worker remains alive. Exit codes: `0` completed or overlap skipped, `1` scraper/worker failure or partial failure, `2` invalid/missing configuration, `124` timeout. Skips leave the previous report intact. Reports and logs accumulate; archive or remove older files as needed.

LinkedIn and other boards can change, block requests, or return empty results. Logged JobSpy errors and exceptions produce a failed/partial report; a silently empty response cannot reliably be distinguished from no matches. Check the log when a source repeatedly returns zero. This command does not bypass sign-in challenges or CAPTCHAs. JobSpy's upstream interface is documented at https://github.com/speedyapply/JobSpy.

For Codex-managed recurring runs, configure a task automation to execute this same command using the project's absolute virtualenv path and summarize new roles or failures from `reports/latest.json`. The automation should stay quiet when no new roles or actionable errors appear.


## Quality-first multi-source discovery

Add to `searches.yaml` to collect curated lists and public sources before broad boards:

```yaml
discovery_sources: [github, public, jobspy]
quality:
  enabled: true
  min_score: 60
  shortlist_limit: 25
feeds:
  - name: We Work Remotely
    url: https://weworkremotely.com/remote-jobs.rss
    remote: true
    company_prefix: true
public_pages: []
```

`public_pages` accepts `{name, url}` entries pointing to public employer job pages containing JSON-LD `JobPosting` data. It is not a site-wide crawler. Login pages and pages without parsable jobs are reported as `empty_or_unreadable`, not treated as successful coverage. RSS expiration and JSON-LD `validThrough` are checked when available. Do not set `remote: true` on a feed unless its jobs are explicitly remote.

Configure selected GitHub tables in `github_repos.yaml`. For example:

```yaml
repos:
  simplifyjobs:
    owner: SimplifyJobs
    repo: New-Grad-Positions
    branch: dev
    file: README.md
    format: html_table
    max_age_days: 30
    columns: {company: 0, title: 1, location: 2, apply: 3, age: 4}
```

The HTML parser reads the first table (currently the software-engineering section), inherits company names on continuation rows, and skips closed listings. Known age above the configured threshold is excluded; unknown age is retained and does not establish freshness. Raw collection and shortlist counts are separate.

The initial quality policy targets new-grad/junior software or founding engineer roles, with explicit remote or NYC location evidence. It rejects senior/staff/management/intern titles and unclear locations. Explicit early-career or founding titles rank above generic software titles; curated new-grad context supports otherwise generic titles. Scores are deterministic matching heuristics, not AI resume scores or verified employer quality. Founding experience requirements, remote eligibility, and posting availability remain review items. Source platform itself gives no ranking bonus, except the explicit new-grad context of a curated list. Reports deduplicate normalized destination URLs within the shortlist and show at most the configured limit of new candidates.

Handshake is not an authenticated collector in this implementation. Public posting URLs can be tried through `public_pages`; account-gated listings require a separate authorized integration. Generic links from any source are not evidence that the full platform is covered.

LinkedIn description fetching is disabled by default for scheduled crawls. Set `defaults.linkedin_fetch_description: true` to opt in to the extra requests. Indeed remote searches omit the conflicting age filter, so their results can be older than `hours_old`; eligibility and freshness still need review. Source failures are recorded independently so other collectors continue.
