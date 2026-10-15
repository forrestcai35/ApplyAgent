# CLI harness for AI stages

ApplyAgent can use a logged-in agent CLI instead of configuring an LLM API key.
This follows Sophic Assistant's spawner pattern: bounded subprocess turns,
explicit context, structured results, isolated working directory, and no automatic
provider failover. The existing `get_client().chat()` interface routes enrichment,
scoring, resume tailoring, cover letters, and their judges through the harness.

Set in your private `APPLYAGENT_DIR/.env`:

```dotenv
APPLYAGENT_LLM_BACKEND=codex
APPLYAGENT_HARNESS_TIMEOUT=180
# Optional; omit to use the CLI default model:
# APPLYAGENT_HARNESS_MODEL=your-model
```

Supported backends: `codex`, `claude`, `api`, `auto`. Auto preserves configured API
providers and otherwise selects Claude. Explicit selection never falls back to an
API or another engine. Authenticate the selected CLI separately. CLI presence
passes dependency checks; it does not prove account access or remaining usage.
Optional executable overrides: `APPLYAGENT_CODEX_BIN`, `APPLYAGENT_CLAUDE_BIN`.

Claude runs with no built-in tools, no MCP servers, and no saved session. Codex
uses `exec`, ephemeral sessions, read-only sandboxing, no shell tool, disabled
apps/web search/multi-agent, and ignored user config. Each task runs in a temporary
directory so project instructions are not loaded. Prompts go through stdin;
results return to the existing parsers and validators. A timeout kills and
reaps the process group. Empty, malformed, and unsuccessful results fail the turn.
Temperature and token-limit arguments are retained for interface compatibility but
are not enforced as CLI sampling controls. Calls require online provider access,
send supplied task data to that provider, and consume the selected account's usage.

```bash
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent run score
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent run tailor cover pdf
```

The browser application launcher supports both Codex and Claude:

```bash
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent apply --backend codex --limit 10
```

`--backend auto` uses `APPLYAGENT_APPLY_BACKEND` when set, otherwise follows the
Codex/Claude text backend. API text configurations prefer an installed Codex CLI,
then Claude. Explicit selection never falls back. `--model` is optional: Codex
uses its default and Claude uses Sonnet. Both need Node/npx and a Chromium browser;
Chrome, Chromium, and macOS Brave are detected. CLI presence does not prove login.
Codex uses its own authenticated CLI session, an isolated browser via Playwright
MCP, no shell tools, and automatic approval review in a worker workspace. Approval requirements are not bypassed;
operations needing unavailable approval stop. The app's connected browser tools
are a separate execution path, not inherited by this CLI subprocess.

Ordinary apply runs replenish an empty queue automatically. The refill path
prioritizes explicit junior roles, checks beyond the first rejected batch, and
uses the original resume for suitable candidates without waiting for tailoring.
It does not relax fit, destination authorization, or mandatory-answer rules.
`--refill-rounds` bounds replenishment (default 3), with at most one public discovery
round per run; `--no-refill` keeps selection within the existing ready queue.
Runs stop when no suitable prospects are prepared or daily capacity is exhausted.
All live modes use one worker and a shared run lock. The profile's
`application_workflow.daily_submission_cap` (default 10) counts submissions from
all runs in its configured timezone (default America/Los_Angeles). Unresolved
in-progress claims block restarting after a supervisor crash.

Codex jobs have a 600-second deadline (`APPLYAGENT_APPLY_TIMEOUT` override). Only
the final message is parsed; success requires a confirmation URL and message.
Timeouts, crashes, and ambiguous outcomes are held as `submission_uncertain`,
without automatic retry. Review their logs before any explicit retry. Dry runs
never count as submissions. Both runners filter the saved application registry
before selecting jobs. Text-generation and unit tests are not submission evidence.

CLI configuration follows [noninteractive Codex documentation](https://developers.openai.com/codex/noninteractive)
and [MCP configuration](https://developers.openai.com/codex/mcp).

Reference: [OpenAI's noninteractive Codex execution example](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).

## Application answer completion

The browser prompt classifies questions as confirmed facts, generated prose,
unambiguous derived facts, optional unknowns, or required unknowns. Motivation,
experience summaries and required cover-letter prose are written from resume
evidence without requesting prewritten answers. Optional unknowns are omitted or
declined. Expected degrees remain expected; date precision, relocation boundaries
and actual employment titles are preserved. Required unknown factual fields remain
blocked, with the exact field and missing evidence recorded.

`profile.json` may include `application_workflow.routine_answer_defaults`, a
dedicated section rendered into both application harnesses. Its boolean switches
are `draft_routine_narratives`, `omit_optional_unknowns`, `decline_optional_sms`,
`onsite_after_approved_relocation` and `willing_to_complete_standard_assessment`.
Office attendance defaults apply only after relocation to a saved destination;
assessment willingness is a future preference, not an assessment answer. Defaults
never establish experience totals, past employment/referrals, remote geography,
document availability or employer/data authorization. Unknown keys and non-boolean
values fail before browser work, preventing arbitrary factual or permission
overrides. Actual discovery sources are used instead of fabricated referrals.

Confirmed screening facts can be saved separately in private
`screening_answers.prior_employment`: each record has `employer_names` (explicit
aliases), boolean `answer`, boolean `includes_subsidiaries`, and the actual
`human_instruction`. Both harnesses receive matching records before deciding a
required answer is unknown. Employer matching is exact after punctuation/case
normalization; no substring matching or blanket negative history. Resume omissions
do not establish exhaustive history. Conflicting answers fail before browser work.
Only the exact hold supported by a fresh confirmation should be reopened; do not
reset unrelated failures or prior tool approval/uncertain-submission holds.

An applicant may grant standing operator consent for routine applications. Record
their actual instruction and standard data scope in the private profile's
`application_workflow.standing_submission_authorization`. The prompt then stops
requesting operator approval for every employer within the saved role/location
scope. Specific grants can also be saved in `authorized_applications`, keyed by
exact employer application URL with the human instruction and data categories.
Neither setting changes tool approval policy, authorizes a rejected action, fills
unknown facts or clears legal/duplicate holds. Old negative cache entries must be
resolved individually from current evidence; do not globally reset failures.

Known answers rejected by form widgets get bounded repairs using fresh page state,
visible options, typing and blur. Explicit validation failures can be retried; an
uncertain post-submit outcome cannot. These changes do not grant destination
authorization, alter tool approvals or guarantee that an employer accepts a form.

## Local integration smoke test

Run `.venv/bin/python scripts/smoke_application_pipeline.py --output-dir /absolute/path/to/fresh-test-directory`
to exercise local discovery, enrichment, configured Codex scoring, browser answer
completion, a synthetic form submission, observed confirmation, receipt persistence
and a daily-cap restart. It uses an isolated synthetic profile and localhost ATS;
no real applicant data, browser cookies or employer applications are used. It
requires the authenticated Codex CLI and browser dependencies and consumes model
usage. A fresh output directory is required, and all test receipts remain there.

Brave enrichment uses an isolated loopback CDP session because its native
Playwright debug-pipe launch can time out. Startup is bounded, cleanup targets only
the owned process, and explicit proxies retain the native launch path.
